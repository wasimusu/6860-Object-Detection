#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image
import numpy as np


def scale_image_channel(im, c, v):
    """
    :param im: Input image
    :param c: Channel on which to perform scaling
    :param v: Scaling factor
    :return: Return scaled image
    """
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    """
    Distort an image
    Step :
        convert to HSV
        Make distortions
        Convert back to RGB
    """
    im = im.convert('HSV')
    cs = list(im.split())  # Split the channels
    cs[1] = cs[1].point(lambda i: i * sat)  # Multiply each pixel by sat
    cs[2] = cs[2].point(lambda i: i * val)  # Multiply each pixel by sat
    
    def change_hue(x):
        """ Scale and shift to adjust the range to 0-255 """
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)  # Change hue of each pixel
    im = Image.merge(im.mode, tuple(cs))  # Merge all the channels back

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    """ Return a random scale """
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    """ Returns a color distorted image """
    dhue = random.uniform(-hue, hue)  # Hue randomly selected
    dsat = rand_scale(saturation)  # Random scale for saturation
    dexp = rand_scale(exposure)  # Random scale for exposure
    res = distort_image(im, dhue, dsat, dexp)  # Change the HSV values
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    """
    :param img: Input image
    :param shape: Crop to this shape
    :param jitter: Jitter in the image dimension
    :param hue: Change hue by this factor
    :param saturation: Change saturation by this factor
    :param exposure: Change exposure by this factor
    :return: return distorted image and factors to adjust the bounding box ground truth
    """
    # Original dimension of image
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    # Compute the cropping coordinate and dimension
    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright   # New width
    sheight = oh - ptop - pbot      # New height

    sx = float(swidth)  / ow        # Change of horizontal scale
    sy = float(sheight) / oh        # Change of vertical scale

    flip = random.randint(1, 10000) % 2  # to flip or not to flip
    cropped = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))  # Cropped image

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)  # Resize the

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)  # Change the color of the cropped image
    
    return img, flip, dx, dy, sx, sy

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    """
    Adjust the ground truth because the input was distorted

    :param labpath:
    :param w: width of image
    :param h: height of image
    :param flip: if the image was flipped
    :param dx: change in x
    :param dy: change in y
    :param sx: factor of horizontal scaling
    :param sy: factor vertical scaling
    :return: new ground truth
    """
    max_boxes = 50
    label = np.zeros((max_boxes,5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    """ Given a path - read images and distort them and return distorted input and target values """
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

    # data augmentation
    # Read image and convert to RGB
    img = Image.open(imgpath).convert('RGB')
    # Distort image
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    # Adjust the label
    label = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
    return img,label
