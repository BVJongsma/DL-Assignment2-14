import pathlib
from PIL import Image, ImageCms
import numpy as np
import os
import cv2
import os
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


def augment_images(path):

    #create directories if they're not already present
    if not os.path.exists('L_landscapes'):
        os.mkdir('L_landscapes')

    if not os.path.exists('L_landscapes/' + 'L'):
        os.mkdir('L_landscapes/' + 'L')
        if len(os.listdir('L_landscapes/L')) > 0:
            print("This directory already contains images.")
            return

    if not os.path.exists('L_landscapes/' + 'a'):
        os.mkdir('L_landscapes/' + 'a')
        if len(os.listdir('L_landscapes/a')) > 0:
            print("This directory already contains images.")
            return

    if not os.path.exists('L_landscapes/' + 'b'):
        os.mkdir('L_landscapes/' + 'b')
        if len(os.listdir('L_landscapes/b')) > 0:
            print("This directory already contains images.")
            return

    #split every image in 3 and save it
    for im in os.listdir(path):
        L, a, b = color_lab(path + '/'+ im)
        L.save('L_landscapes/L/' + im)
        a.save('L_landscapes/a/' + im)
        b.save('L_landscapes/b/' + im)
    return



augment_images("color")



"""
L,a,b = color_lab('../colorwheel.jpg')
L.save('L.png')

images = []
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
        images.append(img)
        """

