"""
Create and save a grid of generated image using a pretrained generator.
"""

import os
import argparse
import torch

from train import folder_tool
from GAN_utils import truncated_gaussian, networks, generate_samples



def make_ims(args, device):

    # Folder
    folder = args.folder
    required_files = [f'generator_{args.generator}_last.pth']
    assert folder_tool(folder, required_files), "folder is empty or doesn't contain a generator"
    img_folder = os.path.join(folder, 'generated_images')
    os.makedirs(img_folder, exist_ok=True)
    
    # Get batch of possibly truncated noise
    latent_dim_dict = {'infogan': 64,
                       'sndcgan': 128}
    loader = truncated_gaussian.truncated_noise(args.num_samples, latent_dim_dict[args.generator], args.noise_bound) # 
    noise_batch = next(loader)[0].to(device)
 
    # Generator network 
    params_g = {}
    params_g['im_res'] = args.resolution
    params_g['num_chan'] = args.channels
    model_g = args.generator + '_g'
    generator = getattr(networks, model_g)(**params_g) 
    # Load generator state dict
    if f'generator_{args.generator}_best.pth' in os.listdir(folder):
        generator.load_state_dict(torch.load(os.path.join(folder, f'generator_{args.generator}_best.pth')))
    else:
        generator.load_state_dict(torch.load(os.path.join(folder, f'generator_{args.generator}_last.pth')))
    generator = generator.to(device)
    
    # Generate images
    im_name = args.im_name
    if f'{im_name}.{args.ext}' in os.listdir(img_folder): # Avoid overwriting previous image
        i=2
        while f'{im_name}_{i}.{args.ext}' in os.listdir(img_folder):
            i += 1
        im_name = f'{im_name}_{i}'
    im_batch = generator(noise_batch)
    generate_samples.generate_image(im_name,
                                    im_batch,
                                    img_folder,
                                    ext=args.ext,
                                    grid_columns=args.grid_columns, 
                                    dpi=args.dpi)






if __name__ == '__main__':
    parser = argparse.ArgumentParser("Creating generated images with pretrained model.")
    
    parser.add_argument('--folder',       type=str,   required=True,        help="Main experiment folder created by running train.py.")
    parser.add_argument('--im_name',      type=str,   default="new_images", help="Name of file under which image is save (minus the file extension)")
    parser.add_argument('--ext',          type=str,   default="jpg",        help="File extension.")
    parser.add_argument('--num_samples',  type=int,   default=100,          help="Number of individual samples to produce.")
    parser.add_argument('--grid_columns', type=int,   default=10,           help="Generated images will be arranged in a grid with this many columns.")
    parser.add_argument('--dpi',          type=int,   default=3000,         help="dpi value for output image. Larger value means higher resolution. Must be > 0")
    parser.add_argument('--noise_bound',  type=float, default=float('inf'), help="Noise bound for truncation trick. Default float('inf') means no truncation.")
    parser.add_argument('--generator',  default='infogan', help="Generator model - must be same as used when running train.py", choices=['infogan','sndcgan'])
    parser.add_argument('--resolution', type=int, default=32, help="Image resolution - should correspond to dataloader used when running train.py.")
    parser.add_argument('--channels',   type=int, default=1,   help="Number of channels - should correspond to dataloader used when running train.py.")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = parser.parse_args()
    
    make_ims(args, device)