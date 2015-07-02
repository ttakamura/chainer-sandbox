import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import time
import argparse
import cPickle as pickle
import os
import sys
import cProfile

# @profile
def forward_one_step(h1, h2, cur_word, next_word, volatile=False):
    word   = V(cur_word, volatile=volatile)
    x      = F.leaky_relu(model.embed(word))

    tmp_x  = model.Wx1(x)
    tmp_h1 = model.Wh1(h1)
    h1     = F.leaky_relu(tmp_x + tmp_h1)

    tmp_x2 = model.Wx2(h1)
    tmp_h2 = model.Wh2(h2)
    h2     = F.leaky_relu(tmp_x2 + tmp_h2)

    y      = model.Wy(h2)
    t      = V(next_word, volatile=volatile)
    loss   = F.softmax_cross_entropy(y, t)
    pred   = F.softmax(y)
    return h1, h2, loss, np.argmax(cuda.to_cpu_async(pred.data))

def init_forward(x_list, volatile=False):
    init_h1 = np.zeros((batch_size, hidden_size), dtype=dt)
    init_h2 = np.zeros((batch_size, hidden_size), dtype=dt)
    if args.gpu >= 0:
        init_h1 = cuda.to_gpu_async(init_h1)
        init_h2 = cuda.to_gpu_async(init_h2)
        x_list  = cuda.to_gpu_async(x_list)
    prev_h1 = V(init_h1, volatile=volatile)
    prev_h2 = V(init_h2, volatile=volatile)
    return prev_h1, prev_h2, x_list

def forward(x_list, volatile=False):
    prev_h1, prev_h2, x_list = init_forward(x_list)
    loss  = 0
    preds = list()
    for cur_word, next_word in zip(x_list, x_list[1:]):
        prev_h1, prev_h2, new_loss, pred = forward_one_step(prev_h1, prev_h2, cur_word, next_word, volatile=volatile)
        loss += new_loss
        preds.append(pred)
    return loss, preds

# @profile
def train(docs, optimizer):
    optimizer.zero_grads()
    loss, results = forward(docs)
    loss.backward()
    optimizer.clip_grads(grad_clip)
    optimizer.update()
    return loss, results

def train_all(epochs, train_docs, optimizer):
    for t in xrange(0, epochs):
        indexes = np.random.permutation(len(train_docs))
        for idx, i in enumerate(indexes):
            docs = np.array(train_docs[i])
            loss, results = train(docs, optimizer)
            if idx % (batch_size * 10) == 0:
                report(docs, t, idx, loss, results)
        serialize_model(args.model, 1, model, vocablary)

def predict(x_list, pred_num=10, volatile=False):
    prev_h1, prev_h2, x_list = init_forward(x_list)
    preds = list()
    pred  = None
    for cur_word in x_list:
        prev_h1, prev_h2, new_loss, pred = forward_one_step(prev_h1, prev_h2, cur_word, cur_word, volatile=volatile)
    for i in xrange(0, pred_num):
        cur_word = np.array([pred])
        if args.gpu >= 0:
            cur_word = cuda.to_gpu_async(cur_word)
        prev_h1, prev_h2, new_loss, pred = forward_one_step(prev_h1, prev_h2, cur_word, cur_word, volatile=volatile)
        preds.append(pred)
    return preds

def parse_text_from_file(text_file, vocab={}, ignore_unknown=False):
    lines = []
    with open(text_file) as f:
        lines = [l.decode('utf-8') for l in f.readlines()]
    return parse_text(lines, vocab, ignore_unknown)

def parse_text(lines, vocab={}, ignore_unknown=False):
    results = []
    for current_line in lines:
        line = []
        if not current_line is None and len(current_line) >= min_length:
            for char in current_line:
                if not char in vocab:
                    if not ignore_unknown:
                        vocab[char] = len(vocab.keys())
                if char in vocab:
                    line.append(vocab[char])
            results.append(line)
    return results, vocab

def to_text(char_list, vocab):
    index = {}
    for k in vocab.keys():
        index[vocab[k]] = k
    return ''.join([index[c] for c in char_list]).encode('utf-8')

def doc_to_minbatch(documents, batch_size):
    batches = []
    for doc in documents:
        batches.append( [[c] for c in doc] )
    return batches

def load_model(namespace, epoch):
    file_path = "data/model_{0}_{1}.pickle".format(namespace, epoch)
    if os.path.exists(file_path):
        model, vocab = pickle.load(open(file_path, 'rb'))
        if args.gpu >= 0:
            model.to_gpu()
        return model, vocab
    else:
        return None

def serialize_model(namespace, epoch, model, vocab):
    file_path = "data/model_{0}_{1}.pickle".format(namespace, epoch)
    pickle.dump((cuda.to_cpu_async(model), vocab), open(file_path, 'wb'), -1)

def report(docs, t, idx, loss, results):
    print "{0} epoch, {1}th data".format(t, idx)
    print loss.data, np.linalg.norm(cuda.to_cpu_async(model.Wh1.W)), np.linalg.norm(cuda.to_cpu_async(model.Wh2.W)), np.linalg.norm(cuda.to_cpu_async(model.Wx1.W))
    print "------------------------------------------"
    print to_text([d[0] for d in docs], vocablary)
    print "=========================================="
    print to_text(results, vocablary)
    print "------------------------------------------"

# --------------------------------------------------------------------------
# ---- init ----------------------------------------------------------------
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu',   '-g', default=-1,         type=int)
parser.add_argument('--text',  '-t', default='paul.txt', type=str)
parser.add_argument('--model', '-m', default='none',     type=str)
parser.add_argument('--mode',        default='train',    type=str)
parser.add_argument('--epoch',       default=10000,      type=int)
args = parser.parse_args()

V = Variable
dt = np.float32

min_length = 2
documents, vocablary = parse_text_from_file(args.text)

vocab_size     = len(vocablary)
batch_size     = 1
embed_size     = 100
hidden_size    = 300
num_train_data = int(len(documents) * 0.8)
learning_rate  = 0.0005
grad_clip      = 10.0
epochs         = args.epoch

train_docs, test_docs = np.split(doc_to_minbatch(documents, batch_size), [num_train_data])

if args.gpu >= 0:
    cuda.init(args.gpu)

if args.mode == 'predict':
    model, vocablary = load_model(args.model, 0)
else:
    model = FunctionSet(
        embed  = F.EmbedID(vocab_size, embed_size),
        Wx1    = F.Linear(embed_size,  hidden_size, wscale=0.1),
        Wx2    = F.Linear(hidden_size, hidden_size, wscale=0.1),
        Wh1    = F.Linear(hidden_size, hidden_size),
        Wh2    = F.Linear(hidden_size, hidden_size),
        Wy     = F.Linear(hidden_size, vocab_size),
    )
    model.Wh1.W = np.eye(model.Wh1.W.shape[0], dtype=dt)
    model.Wh2.W = np.eye(model.Wh2.W.shape[0], dtype=dt)

if args.gpu >= 0:
    model.to_gpu()

# --------------------------------------------------------------------------
# ----- execute ------------------------------------------------------------
# --------------------------------------------------------------------------
if args.mode == 'predict':
    while 1:
        source_text    = sys.stdin.readline().decode('utf-8')
        source_docs, v = parse_text([source_text], vocablary)
        source_data    = np.array( doc_to_minbatch(source_docs, batch_size) )
        results        = predict(source_data[0], 200)
        print to_text(results, vocablary)

else:
    # optimizer = optimizers.AdaGrad(lr=learning_rate)
    optimizer = optimizers.Adam(alpha=learning_rate)
    optimizer.setup(model.collect_parameters())
    if args.mode == 'profile':
        print "Profile mode..............."
        cProfile.run('train_all(epochs, train_docs, optimizer)', 'char_rnn_profile.stats')
    else:
        train_all(epochs, train_docs, optimizer)
