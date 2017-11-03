# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:29:56 2017

@author: deads

References:
    https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    https://docs.opencv.org/trunk/d8/da7/structcv_1_1SimpleBlobDetector_1_1Params.html
    https://docs.opencv.org/3.2.0/d2/d29/classcv_1_1KeyPoint.html

Download and run to count blobs.
Have the appropriate image files in the directory.
"""

# Standard imports
import cv2
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from imagereg import *

def histmatch(im, baseim, basemask):
  oldshape = baseim.shape
  baseim = baseim.ravel()
  basemask = basemask.ravel()
  baseim = baseim[basemask > 0]
  im = im.ravel()
  im = im[basemask > 0]
  sv, bidx, scount = np.unique(baseim, return_inverse=True, return_counts=True)
  tv, tcount = np.unique(im, return_counts=True)
  sq = np.cumsum(scount).astype(np.float64)
  sq /= sq[-1]
  tq = np.cumsum(tcount).astype(np.float64)
  tq /= tq[-1]
  iv = np.interp(sq, tq, tv)
  him = np.zeros(oldshape)
  him = him.ravel()
  him[basemask > 0] = iv[bidx]
  him = him.reshape(oldshape)
  return him

def countHuman(im, baseim, basemask):
  # Warp the image to the base image
  M, tim, _ = registerImage(baseim, im)
  #him = histmatch(tim, baseim, basemask)
  him = tim

  # Setup SimpleBlobDetector parameters.
  params = cv2.SimpleBlobDetector_Params()
  # Change thresholds
  params.minThreshold = 50
  # Filter by Area.
  params.filterByArea = True
  params.minArea = 10
  params.maxArea = 1200
  # Filter by Convexity
  params.filterByConvexity = True
  params.minConvexity = 0.35
  # Allow close blobs
  params.minDistBetweenBlobs = 5.0

  # Create a detector with the parameters
  ver = (cv2.__version__).split('.')
  if int(ver[0]) < 3 :
      detector = cv2.SimpleBlobDetector(params)
  else : 
      detector = cv2.SimpleBlobDetector_create(params)

  # Detect humans with tuned detector
  keypoints = detector.detect(him)
  Blobs_InMask = [(kp.pt[1],kp.pt[0]) for kp in keypoints if basemask[int(kp.pt[1]),int(kp.pt[0])]]

  return len(Blobs_InMask), Blobs_InMask, him


def showResults(im, blobs):
    fig, axes = plt.subplots(1, 1, figsize=(12, 8), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    #ax = axes.ravel()
    for i,(coords,color,title) in enumerate(zip([blobs],['red'],['Predicted'])):
        axes.imshow(im, interpolation='nearest', cmap='gray')
        axes.set_title(title+' Count: '+str(len(coords)))
        for (y,x) in coords:
            c = plt.Circle((x, y), 1, color=color, linewidth=1, fill=False)
            axes.add_patch(c)
    
    plt.tight_layout()
    plt.show()

def saveResults(im, blobs, output):
    fig, axes = plt.subplots(1, 1, figsize=(12, 8), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    #ax = axes.ravel()
    for i,(coords,color,title) in enumerate(zip([blobs],['red'],['Predicted'])):
        axes.imshow(im, interpolation='nearest', cmap='gray')
        axes.set_title(title+' Count: '+str(len(coords)))
        for (y,x) in coords:
            c = plt.Circle((x, y), 1, color=color, linewidth=1, fill=False)
            axes.add_patch(c)
    
    plt.tight_layout()
    plt.savefig(output)
















