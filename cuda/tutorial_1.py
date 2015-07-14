import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.driver   import In, Out

mod = SourceModule("""
__global__ void doublify(float *a)
{
  int idx = threadIdx.x + threadIdx.y*4;
  a[idx] *= 2;
}
""")

#
# ---- manual memcopy -------------------------------------------
#
a_cpu = np.random.randn(4,4).astype(np.float32)
a_gpu = cuda.mem_alloc(a_cpu.nbytes)

cuda.memcpy_htod(a_gpu, a_cpu)

func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))

a_result = np.empty_like(a_cpu)
cuda.memcpy_dtoh(a_result, a_gpu)

print a_result
print a_cpu

#
# ---- auto memcopy -------------------------------------------
#
a_cpu = np.random.randn(4,4).astype(np.float32)

func(cuda.InOut(a_cpu), block=(4,4,1))

print a_cpu

#
# ---- prepared call -------------------------------------------
#
grid  = (1,1)
block = (4,4,1)
func.prepare("P")
func.prepared_call(grid, block, a_gpu)

#
# ---- GPUArray -------------------------------------------
#
b_gpu = gpuarray.to_gpu(np.random.randn(4,4).astype(np.float32))
b_doubled = (2 * b_gpu).get()

print b_gpu
print b_doubled

#
# ---- GPUArray -------------------------------------------
#
b_gpu = gpuarray.to_gpu_async(np.random.randn(100,100).astype(np.float32))
b_doubled = (2 * b_gpu).get_async()

print "This is Async..........."
print b_gpu
print b_doubled
