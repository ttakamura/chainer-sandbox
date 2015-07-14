import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
from pycuda.driver   import In, Out

mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multip_them = mod.get_function("multiply_them")

a = np.random.randn(400).astype(np.float32)
b = np.random.randn(400).astype(np.float32)
dest = np.zeros_like(a)

multip_them(Out(dest), In(a), In(b), block=(400,1,1), grid=(1,1))

print dest - a*b
