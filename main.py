import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from .lib.utils import imread, array_to_tif
from .lib.decon import rl_decon
from timeit import default_timer as timer

import argparse
import time
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]..."
    )

    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('image_dir', nargs='?', const=str)
    parser.add_argument('psf_dir', nargs='?', const=str)
    parser.add_argument('channel', nargs='?', const=str)
    
    
    # time_str = time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime())
    # parser.add_argument('-p', '--prefix', nargs=1, type=str, default=time_str)
    return parser

def decon_dir(args):
    
    otf_dir = args.psf_dir
    channel = args.channel
    psf_file = os.path.join(otf_dir, f'{channel}_psf.tif')
    psf = imread(psf_file)

    start_time = timer()
    print('Decon for  ', args.image_dir)
    root_dir = Path(args.image_dir)
    tiffs = list(root_dir.glob('*.[tT][iI][fF]'))
    decon_dir = os.path.abspath(os.path.join(root_dir.resolve(), f'{root_dir.name}_decon'))
    os.makedirs(decon_dir, exist_ok=True)
    print("tiff_root", root_dir.resolve())

    total_tiffs = len(tiffs)
    for index, tiff in enumerate(tiffs):
        start_img_time = timer()
        target_image = imread(os.path.abspath(os.path.join(root_dir.resolve(), tiff)))
        deconvolved_img = rl_decon(target_image, psf)
        array_to_tif(deconvolved_img, os.path.abspath(os.path.join(decon_dir, Path(tiff).name)))
        print(f'deconvoluted in {timer()-start_img_time}  - {tiff} {index}/{total_tiffs} \n', end='')
        sys.exit(0)
    print(f'\ntotal time for deconvolution {timer()-start_time}')

def main():
    parser = init_argparse()
    args = parser.parse_args()
    decon_dir(args=args)

if __name__ == "__main__":
    main()