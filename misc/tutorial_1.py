import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

x_data = np.array([[1,2,3], [4,5,6]], dtype=np.float32)

x = Variable(x_data)
f = F.Linear(3, 2)
y = f(x)
y.grad = np.ones((2,2), dtype=np.float32)
y.backward()

# ------------------------------------------------------
model = FunctionSet(
    l1 = F.Linear(4, 3),
    l2 = F.Linear(3, 2),
)
x = Variable(np.array([[1,2,3,4], [4,5,6,7]], dtype=np.float32))
h1 = model.l1(x)
h2 = model.l2(h1)
h2.grad = np.ones((2,2), dtype=np.float32)
h2.backward()
