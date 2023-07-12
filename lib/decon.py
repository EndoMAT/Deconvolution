from scipy.signal import fftconvolve, convolve, deconvolve
from scipy import fftpack
from tifffile import imread, imsave
import os
from pathlib import Path
from timeit import default_timer as timer
import numpy as np
from skimage import color, data, restoration
import sys

import configparser



def m_convolve(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    k_shape = np.array(kernel.shape)
    i_shape = np.array(image.shape)
    shape = k_shape+i_shape -1
    image_padded = np.zeros(shape)
    complex_result = (np.issubdtype(image.dtype, np.complexfloating) or
                      np.issubdtype(kernel.dtype, np.complexfloating))
    fshape = [fftpack.helper.next_fast_len(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])

    image_padded[int(kernel.shape[0]/2)-1:-1, int(kernel.shape[1]/2)-1:-1, int(kernel.shape[2]/2)-1:-1] = image
    for z in range(image.shape[2]):  # Loop over every pixel of the image
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                # element-wise multiplication of the kernel and the image
                print(z,y,x)
                print(z+kernel.shape[0], y+kernel.shape[1], x+kernel.shape[2])
                # sys.exit(0)
                output[z, y, x] = (kernel*image_padded[z:z+kernel.shape[0], y:y+kernel.shape[1], x:x+kernel.shape[2]]).sum()
    return output


def rl_decon(image, psf, iterations=10):
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = np.flipud(np.fliplr(psf))

    for _ in range(iterations):
        relative_blur = image / m_convolve(im_deconv, psf)
        im_deconv *= m_convolve(relative_blur, psf_mirror)


'''
def docon_ft(image, psf):
    fft_img = np.fft.fft(image)
    fft_psf = np.fft.fft(psf)
    div_fft = fft_img/fft_psf
    img = np.fft.ifft(div_fft)
    return img

def decon_rl(image, psf, iteration = 10):
    y = image
    h = psf
    x = np.true_divide(h, y)
    print(x.shape)
    ht = h.getH()
    dimg = y
    for i in range(iteration):
        p1 = np.multiply(x, ht)
        p2 = np.divide(y, dimg)
        dimg = np.dot(p1, p2)
        x = np.divide(h, dimg)
    return dimg


def rl_decon(image, psf, iteration = 10):
    fft_image = np.fft.fftn(image.astype(np.float64))
    psf = np.resize(psf.astype(np.float64), image.shape)
    fft_psf = np.fft.fftn(psf.astype(np.float64))
    im_deconv = fft_image
    # for i in range(iteration):
    imd = im_deconv / fft_psf

    imd = np.fft.ifftn(imd)
    imd = imd.astype(np.float)
    return imd

def richardson_lucy(image, psf, iterations=10, clip=False):
    """Richardson-Lucy deconvolution.

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.

    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.

    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> camera = color.rgb2gray(data.camera())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> camera = convolve2d(camera, psf, 'same')
    >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """

    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = np.fliplr(psf)

    for _ in range(iterations):
        relative_blur = image / fftconvolve(im_deconv, psf, 'same')
        im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')

    # if clip:
    #     im_deconv[im_deconv > 1] = 1
    #     im_deconv[im_deconv < -1] = -1

    return im_deconv
'''