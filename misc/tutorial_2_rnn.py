import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import time

vocab_size = 80
V = Variable
dt = np.float32

model = FunctionSet(
    embed  = F.EmbedID(vocab_size, 100),
    Wx     = F.Linear(100, 50),
    Wh     = F.Linear( 50, 50),
    Wy     = F.Linear( 50, vocab_size),
)
optimizer = optimizers.SGD()
optimizer.setup(model.collect_parameters())

def forward_one_step(h, cur_word, next_word, volatile=False):
    word = V(cur_word, volatile=volatile)
    t    = V(next_word, volatile=volatile)
    x    = F.tanh(model.embed(word))
    h    = F.tanh(model.Wx(x) + model.Wh(h))
    y    = model.Wy(h)
    loss = F.softmax_cross_entropy(y, t)
    pred = F.softmax(y)
    return h, loss, np.argmax(pred.data)

def forward(x_list, volatile=False):
    prev_h = V(np.zeros((1,50), dtype=dt), volatile=volatile)
    loss   = 0
    preds  = list()
    for cur_word, next_word in zip(x_list, x_list[1:]):
        prev_h, new_loss, pred = forward_one_step(prev_h, cur_word, next_word, volatile=volatile)
        loss += new_loss
        preds.append(pred)
    return loss, preds

for t in xrange(0, 10):
    document = np.array([[5], [4], [3], [2], [1], [2], [3], [4], [5]])
    optimizer.zero_grads()
    loss, results = forward(document)
    print document, results
    loss.backward()
    optimizer.update()

print "OK!"
