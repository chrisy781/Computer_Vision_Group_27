#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:25:30 2022

@author: floriskrijgsman
"""
import pandas as pd
import glob
import argparse
import os
from shutil import copyfile

def tiler(imnames, newpath, falsepath, slice_size, ext):
    a = 0
    for imname in imnames:
        labname = imname.replace(ext, '.txt')
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
        diff_loc = (labels['x1']-labels['y1'])[0]
        diff_dims = (labels['w']-labels['h'])[0]
        if diff_loc == 0:
            if diff_dims == 0:
                print("faulty", imname)
                os.remove(imname)
                os.remove(labname)
                print("both files removed")
#                time.sleep(10)
                a += 1
        
    print("amount of mistakes", a)

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-source", default="./obj/", help = "Source folder with images and labels needed to be tiled")
    parser.add_argument("-target", default="./new_tiled/", help = "Target folder for a new sliced dataset")
    parser.add_argument("-ext", default=".jpg", help = "Image extension in a dataset. Default: .JPG")
    parser.add_argument("-falsefolder", default=None, help = "Folder for tiles without bounding boxes")
    parser.add_argument("-size", type=int, default=100, help = "Size of a tile. Default: 416")
    parser.add_argument("-ratio", type=float, default=1, help = "Train/test split ratio. Dafault: 0.8")

    args = parser.parse_args()

    imnames = glob.glob(f'{args.source}/*{args.ext}')
    labnames = glob.glob(f'{args.source}/*.txt')
    if len(imnames) == 0:
        raise Exception("Source folder should contain some images")
    elif len(imnames) != len(labnames):
        raise Exception("Dataset should contain equal number of images and txt files with labels")

    if not os.path.exists(args.target):
        os.makedirs(args.target)
    elif len(os.listdir(args.target)) > 0:
        raise Exception("Target folder should be empty")

    # classes.names should be located one level higher than images   
    # this file is not changing, so we will just copy it to a target folder 
    upfolder = os.path.join(args.source, '..' )
    target_upfolder = os.path.join(args.target, '..' )
    if not os.path.exists(os.path.join(upfolder, 'classes.names')):
        print('classes.names not found. It should be located one level higher than images')
    else:
        copyfile(os.path.join(upfolder, 'classes.names'), os.path.join(target_upfolder, 'classes.names'))
    
    if args.falsefolder:
        if not os.path.exists(args.falsefolder):
            os.makedirs(args.falsefolder)
        elif len(os.listdir(args.falsefolder)) > 0:
            raise Exception("Folder for tiles without boxes should be empty")

    labels = tiler(imnames, args.target, args.falsefolder, args.size, args.ext)
    splitter(args.target, target_upfolder, args.ext, args.ratio)