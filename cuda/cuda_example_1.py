import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.driver   import In, Out
