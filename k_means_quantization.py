
# create a looup table including: 
# 1. the centroids - 2**n, in fp32 
# 2. labels: n-bit integer tensor, same number of elements as the orginal weight tensor 


# create lookup table 

from collections import namedtuple
from fast_pytorch_kmeans import KMeans

LookupTable = namedtuple('LookupTable', ['centroids, labels'])

def k_means_quantize(fp32_tensor : torch.Tensor, bitwidth=4, lookuptable = None):
    pass 