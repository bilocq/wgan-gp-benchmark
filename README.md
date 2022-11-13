# WGAN-GP Benchmark
This repository is a PyTorch implementation of WGAN-GP (Gulrajani et al. 2017) that was used to run the benchmark image generation experiments in [*A new method for determining Wasserstein 1 optimal transport maps from Kantorovich potentials, with deep learning applications*](https://arxiv.org/abs/2211.00820). The main codebase for the TTC algorithm introduced in that paper is [here](https://github.com/bilocq/Trust-the-Critics-2).


## How to run this code ##
* Create a Python virtual environment with Python 3.8 installed.
* Install the packages listed in requirements.txt (pip install -r requirements.txt)
* Training and evaluating performance through FID are both done by running the train.py script. The results will be saved in a directory specified by the --folder argument (this directory should either not exist or be empty when running train.py).

The image generation experiments we ran with WGAN-GP are described in Section 5 and Appendix 8 of the paper. Here, we include an example of a shell script that could be used (after minor modifications) to run WGAN-GP experiments. Running train.py will train an instance of WGAN-GP for a specified number of iterations while also stopping training at regular intervals to evaluate performance through FID. The number of FID evaluations is specified through the --checkpoints argument. A past experiment can be resumed by specifying the same path with the --folder argument. Note that training WGAN-GP is computationally demanding, and thus requires adequate computational resources (i.e. running this on a regular laptop is not recommended).


### Computing architecture and running times
We ran the code presented here on computational clusters provided by the Digital Research Alliance of Cananda (https://alliancecan.ca/en), always using a single NVIDIA P100 or V100 GPU. Training times are reported in Addendix 8 of the paper.


## Assets 
Portions of this code, as well as the datasets used to produce our experimental results, make use of existing assets. We provide here a list of all assets used, along with the licenses under which they are distributed, if specified by the originator:
- This codebase was built from a PyTorch implementation (https://github.com/caogang/wgan-gp) of WGAN-GP ((c) 2017 Ishaan Gulrajani).
- **pytorch_fid**: from https://github.com/mseitzer/pytorch-fid. Distributed under the Apache License 2.0.
- **MNIST dataset**: from http://yann.lecun.com/exdb/mnist/. Distributed under the Creative Commons Attribution-Share Alike 3.0 license.
- **Fashion MNIST datset**: from  https://github.com/zalandoresearch/fashion-mnist ((c) 2017 Zalando SE, https://tech.zalando.com). Distributed under the MIT licence.
- **CelebA-HQ dataset**: from https://paperswithcode.com/dataset/celeba-hq
- **CIFAR10 dataset**: from https://www.cs.toronto.edu/~kriz/cifar.html.
- **FFHQ dataset**: from https://github.com/NVlabs/ffhq-dataset
