import pathlib
from PIL import Image, ImageCms
import numpy as np


def color_lab(image):
    im = Image.open(image).convert('RGB')
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p = ImageCms.createProfile("LAB")

    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = ImageCms.applyTransform(im, rgb2lab)
    # lab is the image in colourspace

    L, a, b = Lab.split()

    # L.save('L.png')
    # a.save('a.png')
    # b.save('b.png')

    return L, a, b

L,a,b = color_lab('../colorwheel.jpg')
L.save('L.png')