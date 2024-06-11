
# create a looup table including: 
# 1. the centroids - 2**n, in fp32 
# 2. labels: n-bit integer tensor, same number of elements as the orginal weight tensor 


# create lookup table 

from collections import namedtuple
from fast_pytorch_kmeans import KMeans
import torch 

LookupTable = namedtuple('LookupTable', ['centroids, labels'])

def k_means_quantize(fp32_tensor : torch.Tensor, bitwidth=4, lookuptable = None):
    if lookuptable is None: 
        n_clusters = 0 
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1,-1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        lookuptable = LookupTable(centroids, labels)    

    
    #need to do: decode the quantized tensor 

