"""Code to sample from a truncated Gaussian"""

import torch

def truncated_gaussian(num_samp, bound = 1):
    """Inputs
    -num_samp, number of samples from truncated Gaussian
    -bound, the bound on the Gaussian. Any values over
    this number in absolute value will be resampled. Something near 2.5-3 works well. 
    
    Output
    - a list of size num_samp*1 of random variables sampled
    from truncated Gaussian"""

    samples = torch.Tensor([])
    count = 0
    num_per_batch = num_samp#number of points to generate each iteration

    while samples.size(0) < num_samp:
        x = torch.randn(num_per_batch)
        valid = (x > -bound) & (x < bound)
        samples = torch.cat((samples, x[valid]), dim = 0)
        count += 1
        num_per_batch = max(num_samp - samples.size(0), 100000)
    samples = samples[0:num_samp]
    return samples


def truncated_noise(num_samples, dim, bound):
    nl = TruncNoiseLoader()
    nl.bs = num_samples
    nl.dim = dim
    nl.bound = bound
    nl.num_samp = nl.bs*nl.dim

    return nl

class TruncNoiseLoader:
    def __iter__(self):
        return self

    def __next__(self):
        output = truncated_gaussian(self.num_samp, self.bound).view(self.bs, self.dim)
        return output, None # second is dummy variable