import argparse
import numpy as np
import scipy as sp
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from chainer import cuda
import skchainer as skc

document = list("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 200)
# cuda.init()

def parse(words, vocab):
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset

doc_len = len(document)
vocab   = {}
dataset = parse(document, vocab)

X = dataset[0:doc_len-1].reshape(doc_len-1, 1)
y = dataset[1:doc_len]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# -----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--net_type', type=str, default='irnn')
parser.add_argument('--search', type=str,   default='grid')
parser.add_argument('--n_jobs', type=int,   default=-1)
parser.add_argument('--n_iter', type=int,   default=10)
parser.add_argument('--gpu',    type=int,   default=-1)
args = parser.parse_args()

if args.gpu >= 0:
    cuda.init()

model = skc.RNNCharEstimator(epochs=1, batch_size=10, vocab_size=len(vocab), threshold=1e-6, gpu=args.gpu)

if args.search == 'grid':
    tuned_parameters = [
        {'net_type': ['irnn'], 'opt_type': ['adam', 'adagrad'], 'net_hidden': [100, 200, 300]}
    ]
    skc.grid_search(model, tuned_parameters, X_train, y_train, X_test, y_test, score='accuracy', n_jobs=args.n_jobs)

elif args.search == 'random':
    tuned_parameters = {
        'net_type': ['irnn'], 'opt_type': ['adam', 'adagrad'], 'net_hidden': sp.stats.norm(300, 100)
    }
    skc.random_search(model, tuned_parameters, X_train, y_train, X_test, y_test, score='accuracy', n_jobs=args.n_jobs, n_iter=args.n_iter)

# predict -----------------------
# print model.predict(X[1:5,])
