import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import time
import math
from sklearn.datasets import fetch_mldata

V  = Variable
dt = np.float32

batch_size     = 200
hidden_size    = 100
num_train_data = 60000
num_test_data  = 10000
learning_rate  = 0.0001
grad_clip      = 1.0

mnist = fetch_mldata('MNIST original')
x_all = mnist.data.astype(dt) / 255
y_all = mnist.target.astype(np.int32)
x_train, x_test = np.split(x_all, [num_train_data])
y_train, y_test = np.split(y_all, [num_train_data])

# --------------------------------------------------------------------------------------------

def forward_one_step(h, x_val):
    x  = V(x_val)
    wx = model.Wx(x)
    wh = model.Wh(h)
    h  = F.leaky_relu(wx + wh)
    return h

def forward(x_list, result, use_gpu=False):
    init_h = np.zeros((batch_size,hidden_size), dtype=dt)

    if use_gpu:
        x_list = cuda.to_gpu(x_list)
        result = cuda.to_gpu(result)
        init_h = cuda.to_gpu(init_h)

    prev_h = V(init_h)
    for i in xrange(0, 784):
        x = x_list[:,i]
        prev_h = forward_one_step(prev_h, x)

    t        = V(result)
    y        = model.Wy(prev_h)
    loss     = F.softmax_cross_entropy(y, t)
    accuracy = F.accuracy(y, t)
    return loss, accuracy

# -------------------- model -------------------------------------------
model = FunctionSet(
    Wx     = F.Linear(           1, hidden_size, wscale=0.01),
    Wh     = F.Linear( hidden_size, hidden_size, wscale=0.01),
    Wy     = F.Linear( hidden_size,          10, wscale=0.01),
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

# optimizer = optimizers.SGD(lr=learning_rate)
# optimizer = optimizers.AdaGrad(lr=learning_rate)
optimizer = optimizers.Adam(alpha=learning_rate)
optimizer.setup(model.collect_parameters())

for t in xrange(0, 1000):
    indexes  = np.random.permutation(num_train_data)
    loss     = None
    accuracy = None

    for i in xrange(0, num_train_data, batch_size):
        x_batch = x_train[indexes[i : i + batch_size]]
        y_batch = y_train[indexes[i : i + batch_size]]

        optimizer.zero_grads()
        loss, accuracy = forward(x_batch, y_batch, use_gpu=(args.gpu >= 0))
        loss.backward()
        optimizer.clip_grads(grad_clip)
        optimizer.update()

        if i % (batch_size * 10) == 0:
            print loss.data, accuracy.data, np.linalg.norm(cuda.to_cpu(model.Wx.W)), np.linalg.norm(cuda.to_cpu(model.Wh.W)), np.linalg.norm(cuda.to_cpu(model.Wy.W))

    print "=================================="
    print t
    print loss.data, accuracy.data # , np.sum(model.Wx.W), np.sum(model.Wh.W), np.sum(model.Wy.W)

print "OK!"
