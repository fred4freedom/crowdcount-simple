#!/usr/bin/env python
import numpy as np
import cv2
import argparse
import json

def registerImage(baseim, newim, replicateBorder=True):
    """
    Register the image `newim` to the base image `baseim` where both images are numpy arrays
    """

    # Initialize SIFT detector
    sift = cv2.SIFT()

    # Find the keypoints and descriptors
    kp1, feat1 = sift.detectAndCompute(baseim, None)
    kp2, feat2 = sift.detectAndCompute(newim, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(feat1,feat2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w,c = baseim.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    #tim = cv2.polylines(newim,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    if replicateBorder:
        tim = cv2.warpPerspective(newim, M=M, dsize=(w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)
    else:
        tim = cv2.warpPerspective(newim, M=M, dsize=(w, h), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LINEAR, borderValue=0)

    return M, tim, mask


def transformImage(im, M, replicateBorder=True):
    """
    Transform the image 'im' given the transformation 'M'
    """

    h,w,c = im.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    #tim = cv2.polylines(newim,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    if replicateBorder:
        tim = cv2.warpPerspective(im, M=M, dsize=(w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)
    else:
        tim = cv2.warpPerspective(im, M=M, dsize=(w, h), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LINEAR, borderValue=0)

    return tim



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("other_image")
    parser.add_argument("base_image")
    parser.add_argument("output_image")
    parser.add_argument("--transform", help="file to save the transformation matrix")
    parser.add_argument("--mask", help="file to save the transformation mask")
    args = parser.parse_args()

    base_im = cv2.imread(args.base_image)
    other_im = cv2.imread(args.other_image)
    out_imfn = args.output_image

    M, tim, mask = registerImage(base_im, other_im)
    cv2.imwrite(out_imfn, tim)

    if args.mask:
        cv2.imwrite(args.mask, mask)

    if args.transform:
        with open(args.transform, 'w') as f:
            out = {'transform': M}
            json.dump(out, f)

if __name__ == "__main__":
    main()
