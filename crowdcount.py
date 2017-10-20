#!/usr/bin/env python
import numpy as np
import cv2
import argparse
import json
import os
from imagereg import *
from itertools import *
from scipy.stats import ks_2samp
from skimage.filters import threshold_local
from skimage.exposure import equalize_adapthist


def findBackground(imlist):
    """
    Try to find the background image given a list of images
    """
    idxlist = range(len(imlist))
    bglist = []
    countlist = []
    Mlist = []
    for k in idxlist:
        base_im = imlist[k]
        base_im = (equalize_adapthist(base_im) * 255).astype('uint8')
        base_im_lab = cv2.cvtColor(base_im, cv2.COLOR_BGR2LAB)
        bgim = np.zeros(base_im.shape)
        countim = np.zeros(base_im.shape)
        curMlist = []
        for p in idxlist:
            if p == k:
                curMlist.append(None)
                continue
            curim = imlist[p]
            curim = (equalize_adapthist(curim) * 255).astype('uint8')
            M, ctim, cmask = registerImage(base_im, curim, replicateBorder=False)
            diffim = cv2.cvtColor(ctim, cv2.COLOR_BGR2LAB)
            diffim = np.abs(diffim.astype('float32') - base_im.astype('float32'))
            diffim = np.sum(diffim, axis=2)
            threshold = np.median(diffim) #+ 0.5 * np.std(diffim)
            cbgmask = diffim < threshold
            cbgmask = np.dstack([cbgmask] * base_im.shape[2])
            bgim = bgim + cbgmask * base_im
            countim = countim + cbgmask
            curMlist.append(M)

        bglist.append(bgim)
        countlist.append(countim)
        Mlist.append(curMlist)

    fbglist = []
    for k in idxlist:
        base_im = bglist[k].copy().astype('float32')
        count_im = countlist[k].copy().astype('float32')
        curMlist = Mlist[k]
        for p in idxlist:
            if p == k:
                continue
            curbg = bglist[p]
            curcount = countlist[p] 
            M = curMlist[p]
            curbg = cv2.warpPerspective(curbg, M=M, dsize=(base_im.shape[1], base_im.shape[0]), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LINEAR, borderValue=0)
            curcount = cv2.warpPerspective(curcount, M=M, dsize=(base_im.shape[1], base_im.shape[0]), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LINEAR, borderValue=0)
            base_im = curbg + base_im
            count_im = count_im + curcount
            

        count_im[count_im <= 0] = 1.0
        base_im = base_im / count_im
        base_im = base_im.astype('uint8')
        fbglist.append(base_im)


    return fbglist


def countCrowd(im):
    import skimage
    from skimage.measure import label, regionprops
    from skimage.morphology import square, closing
    from scipy.stats import mode

    im = im > 0

    label_image = label(im)
    num_labels = np.max(label_image)
    props = regionprops(label_image)

    heightlist = [k['bbox'][2] - k['bbox'][0] for k in props]
    heightlist = [k for k in heightlist if k > 1]

    widthlist = [k['bbox'][3] - k['bbox'][1] for k in props]
    widthlist = [k for k in widthlist if k > 1]

    avgpersonheight = np.median(heightlist)
    minheight = 0.3 * avgpersonheight
    maxheight = 1.5 * avgpersonheight

    avgpersonwidth = np.median(widthlist)


    arealist = [
        k['area'] for k in props 
        if k['bbox'][2] - k['bbox'][0] >= minheight and 
        k['bbox'][2] - k['bbox'][0] <= maxheight and 
        k['area'] > 0.75 * avgpersonwidth * avgpersonheight
    ]

    personarea = mode(arealist)[0] / 2.0

    totalcount = 0
    for k in props:
        height = k['bbox'][2] - k['bbox'][0]
        numrows = height / (avgpersonheight / 2.0)
        count = k['area'] / personarea
        count = (numrows - 1) / numrows * count
        totalcount += count


    return totalcount


def filterHuman(im):
    import skimage
    from skimage.measure import label, regionprops
    from skimage.morphology import square, closing, opening
    from scipy.stats import mode

    oim = im.copy()
    im = im > 0

    label_image = label(im)
    num_labels = np.max(label_image)
    props = regionprops(label_image)

    widthlist = [k['minor_axis_length'] for k in props]
    heightlist = [k['major_axis_length'] for k in props]

    strsz = int(max(np.mean(widthlist), np.mean(heightlist)))
    im = closing(im, square(strsz) )
    im = opening(im, square(3))

    label_image = label(im)
    num_labels = np.max(label_image)
    props = regionprops(label_image)
    width_lowthresh = np.median(widthlist) / 2.0
    width_highthresh = np.median(widthlist) * 2.0
    width_higherthresh = np.median(widthlist) * 3.0
    height_lowthresh = np.median(heightlist) / 2.0
    height_highthresh = np.median(heightlist) * 2.0
    height_higherthresh = np.median(heightlist) * 3.0

    labels = label_image
    for k in range(num_labels):
        height = props[k]['major_axis_length']
        width = props[k]['minor_axis_length']

        if width < width_lowthresh or height < height_lowthresh \
           or (width > width_highthresh and width < width_higherthresh) \
           or (height > height_highthresh and height < height_higherthresh) \
           or width <= 0 or height <= 0:
            labels[labels == props[k]['label']] = -1
    
    labels = labels + 1
    labels = (labels > 0).astype('float32') * (im > 0).astype('float32') 
    labels = (255 * labels).astype('uint8')
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--prefix", help="prefix for output background images", default='bg_')
    parser.add_argument("--prefix_fg", help="prefix for output background images", default='fg_')
    args = parser.parse_args()

    prefix = args.prefix
    prefixfg = args.prefix_fg

    inputdir = args.input_dir
    filelist = os.listdir(inputdir)
    imlist = []
    imfnlist = []
    for fn in filelist:
        try:
            print("Reading {}".format(fn))
            im = cv2.imread(os.path.join(inputdir, fn))
            imlist.append(im)
            imfnlist.append(fn)
        except:
            print("Skipping {}".format(fn))

    bglist = findBackground(imlist)
    for fn, bg in zip(imfnlist, bglist):
        fn = prefix + fn
        print("Writing background {}".format(fn))
        cv2.imwrite(os.path.join(args.output_dir, fn), bg)

    # Get foreground
    masklist = []
    for im, bg, fn in zip(imlist, bglist, imfnlist):

        im = (equalize_adapthist(im) * 255).astype('uint8')
        bg = bg.astype('float32') / np.max(bg) * np.max(im)
        diffim = np.abs(im.astype('float32') - bg.astype('float32'))

        mask = np.mean(bg, axis=2)
        mask = mask > 0.5
        mask = np.dstack([mask] * im.shape[2])
        diffim = diffim * mask.astype('float32')
        diffim = np.sum(diffim, axis=2)
        diffim = diffim.astype('float32')
        diffim = (diffim - np.min(diffim[diffim > 0])) / (np.max(diffim[diffim > 0]) - np.min(diffim[diffim > 0]))

        mask = diffim

        thresh = 0.4
        mask = diffim > thresh
        mask = filterHuman(mask)
        count = countCrowd(mask)
        masklist.append(mask)
        mask = np.dstack([mask] * im.shape[2])
        fgim = np.zeros(im.shape)
        fgim[mask > 0] = im[mask > 0]

        fn = prefixfg + fn
        print("Writing foreground {}".format(fn))
        cv2.imwrite(os.path.join(args.output_dir, fn), fgim)
        print("Crowd count = {}".format(count))

        rim = im - fgim
        fn = "res_" + fn
        cv2.imwrite(os.path.join(args.output_dir, fn), rim)

        fn = "diff_" + fn
        cv2.imwrite(os.path.join(args.output_dir, fn), (diffim * 255).astype('uint8'))        



if __name__ == "__main__":
    main()