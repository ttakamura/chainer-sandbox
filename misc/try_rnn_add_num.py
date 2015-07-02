import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import time
import math

V  = Variable
dt = np.float32

batch_size  = 300
seq_size    = 200
hidden_size = 100

def toy_data():
    y_list = []
    x_list = []

    for i in xrange(0, seq_size):
        x_list.append([])

    for j in xrange(0, batch_size):
        a =     int(np.random.rand() * 0.9 * (seq_size / 2))
        b = a + int(np.random.rand() * 0.9 * (seq_size / 2))
        y = 0.0
        for i in xrange(0, seq_size):
            x = np.random.rand()
            z = 0.0
            if i == a or i == b:
                y += x
                z = 1.0
            x_list[i].append([x,z])
        y_list.append([y])

    return np.array(x_list, dtype=dt), np.array(y_list, dtype=dt)

def forward_one_step(h, x_val):
    x  = V(x_val)
    wx = model.Wx(x)
    wh = model.Wh(h)
    h  = F.leaky_relu(wx + wh)
    return h

def forward(x_list, result, use_gpu=False):
    init_h = np.zeros((batch_size,hidden_size), dtype=dt)
    if use_gpu:
        init_h = cuda.to_gpu(init_h)
    prev_h = V(init_h)

    for x in x_list:
        if use_gpu:
            x = cuda.to_gpu_async(x)
        prev_h = forward_one_step(prev_h, x)

    if use_gpu:
        result = cuda.to_gpu_async(result)

    t = V(result)
    y = model.Wy(prev_h)
    loss = F.mean_squared_error(y, t)
    return loss, y

# -------------------- model -------------------------------------------
model = FunctionSet(
    Wx     = F.Linear(           2, hidden_size, wscale=0.01),
    Wh     = F.Linear( hidden_size, hidden_size, wscale=0.01),
    Wy     = F.Linear( hidden_size,           1, wscale=0.01),
)
model.Wh.W = np.eye(model.Wh.W.shape[0], dtype=dt)

# -------------------- execute -------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int)
args = parser.parse_args()

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

optimizer = optimizers.SGD(0.01)
# optimizer = optimizers.AdaGrad(lr=0.01)
optimizer.setup(model.collect_parameters())

for t in xrange(0, 10000):
    xtst, ytst = toy_data()

    optimizer.zero_grads()
    loss, result = forward(xtst, ytst, use_gpu=(args.gpu >= 0))

    loss.backward()
    optimizer.clip_grads(10)
    optimizer.update()

    if t % 10 == 0:
        a = cuda.to_cpu(result.data)[0]
        b = cuda.to_cpu(ytst)
        c = a - b
        print loss.data, np.mean(np.abs(c)) # , np.sum(model.Wx.W), np.sum(model.Wh.W), np.sum(model.Wy.W)

print "OK!"
