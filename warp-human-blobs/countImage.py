#!/usr/bin/env python
import argparse
from humancount import *
from scipy.misc import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--refimage", default='./data/ref.jpg')
    parser.add_argument("--refmask", default='./data/mask.jpg')
    parser.add_argument("--showresult", default=0)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    im = imread(args.image)
    baseim = imread(args.refimage)
    basemask = imread(args.refmask)
    cnt, blobs, him = countHuman(im, baseim, basemask)
    print(cnt)
    if args.save:
      saveResults(him, blobs, args.save)
    if args.showresult:
      showResults(him, blobs)
    return 0

if __name__ == "__main__":
  main()
