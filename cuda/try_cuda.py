import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void cu_multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multiply_them = mod.get_function('cu_multiply_them')

epoch = 10000
a     = np.random.randn(100).astype(np.float32)
b     = np.random.randn(100).astype(np.float32)

for i in xrange(0, epoch):
    d = np.zeros_like(a)
    block = (100,1,1)
    grid = (1,1)
    multiply_them(drv.Out(d), drv.In(a), drv.In(b), block=block, grid=grid)

print d - (a*b)
