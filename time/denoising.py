import numpy as np
from pypfm import PFMLoader
from PIL import Image
import os

HEIGHT = 720
WIDTH = 1280

def tensor_to_pfm(img, name, rescale=False):
    """converts a tensor to a pfm and saves it, can also take othe format than tensor as input

    Args:
        img (Tensor): the image to save
        name (String): the name to give to the stored image
        rescale (bool, optional): whether or not the image should be normalized. Defaults to False.
    """
    img = r(np.array(img, dtype=np.float32))[-HEIGHT:, -WIDTH:, :]
    if rescale:
        img = img / 255.0
    shape = img.shape[1:]
    loader = PFMLoader((shape[1], shape[0]), True, compress=False)
    loader.save_pfm("/home/ascardigli/RL_PATH_TRACING/tmp/" + name + ".pfm", img)


def path_to_pfm(path, name):
    """directly converts the png data from disk to pfm format

    Args:
        path (String): the path of the input image
        name (String): the name of the output image
    """
    img = Image.open(path)
    tensor_to_pfm(img, name, True)


def pfm_to_tensor(path):
    """converts a pfm image to a tensor

    Args:
        path (String): the input path pointing to the pfm image

    Returns:
        Tensor: the converted tensor image
    """
    loader = PFMLoader(color=True, compress=False)
    image = loader.load_pfm(path + ".pfm")
    return image


def denoiser(color, alb, nrm, pid):
    """Calls OIDN denoiser given path to input data and additional data

    Args:
        color (String): path to pfm input image
        alb (String): path to pfm albedo feature
        nrm (String): path to pfm normal feature
        pid (int): specifies the path of the output image 
    """
    a = "/home/ascardigli/RL_PATH_TRACING/"
    os.system(
        "~/oidn-1.4.3.x86_64.linux/bin/./oidnDenoise --ldr "
        + a
        + "tmp/"
        + color
        + ".pfm --alb  "
        + a
        + "tmp/"
        + alb
        + ".pfm --nrm "
        + a
        + "tmp/"
        + nrm
        + ".pfm -v 0 -o "
        + a
        + "tmp/"
        + pid
        + ".pfm > /dev/null"
    )


def denoise(img, pid):
    """denoises an image using OIDN

    Args:
        img (Tensor): color image to be denoised
        pid (int): specifies name of output

    Returns:
        _type_: _description_
    """
    tensor_to_pfm(img, "color")
    denoiser("color", "albedo", "normal", pid)
    return pfm_to_tensor("/home/ascardigli/RL_PATH_TRACING/tmp/" + pid)
