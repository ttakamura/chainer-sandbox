import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
from sklearn.datasets import fetch_mldata
import time

mnist = fetch_mldata('MNIST original')
x_all = mnist.data.astype(np.float32) / 255
y_all = mnist.target.astype(np.int32)
x_train, x_test = np.split(x_all, [60000])
y_train, y_test = np.split(y_all, [60000])

# --------------------------------------------------------------------------------------------

model = FunctionSet(
    l1 = F.Linear(784, 100),
    l2 = F.Linear(100, 100),
    l3 = F.Linear(100,  10),
)
optimizer = optimizers.SGD()
optimizer.setup(model.collect_parameters())

def forward(x_data, y_data):
    x  = Variable(x_data)
    t  = Variable(y_data)
    h1 = F.relu(model.l1(x))
    h2 = F.relu(model.l2(h1))
    y  = model.l3(h2)
    return F.softmax_cross_entropy(y, t), y, t

start = time.time()

batchsize = 100
for epoch in xrange(10):
    print 'epoch', epoch
    indexes = np.random.permutation(60000)
    for i in xrange(0, 60000, batchsize):
        x_batch = x_train[indexes[i:i+batchsize]]
        y_batch = y_train[indexes[i:i+batchsize]]
        # start
        optimizer.zero_grads()
        loss, y, t = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        # report
        if i == 0:
            print F.accuracy(y, t).data

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time))

sum_loss, sum_accuracy = 0, 0
for i in xrange(0, 10000, batchsize):
    x_batch      = x_test[i : i+batchsize]
    y_batch      = y_test[i : i+batchsize]
    loss, y, t   = forward(x_batch, y_batch)
    accuracy     = F.accuracy(y, t)
    sum_loss     += loss.data * batchsize
    sum_accuracy += accuracy.data * batchsize

mean_loss     = sum_loss / 10000
mean_accuracy = sum_accuracy / 10000
print("Mean accuracy: {0}".format(mean_accuracy))
