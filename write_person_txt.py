#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:59:26 2019

@author: fengy
"""
import sys
import os
import argparse
import shutil
import h5py
import numpy as np
import pandas as pd
import scipy.misc as sp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import xml.etree.ElementTree as ET
import glob




def write_to_file(input_path,output_path,vid_list):
        for segment in vid_list:
            fname = os.path.join(output_path, segment) + ".txt"
            print("Writing to {}".format(fname))
            for video in vid_list[segment]:
                path=video.split('.')
                path1=path[0].split('/')
                
                with open(fname,"a+") as f:
                         f.write(os.path.join(path1[-3],path1[-2],path1[-1] + " 1\n"))
                         print(os.path.join(path1[-3],path1[-2],path1[-1] + " 1\n"))






if __name__ == "__main__":

    output_path='/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/ImageSets/DET'
    #input_path='/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/Data/DET/train/ILSVRC2013_train/n00007846'
    input_path='/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/Data/DET/train/FIGURE_EIGHT/n00007846'
    val_input_path='/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/Data/DET/val/FIGURE_EIGHT/n00007846'
    
    videos = glob.glob(os.path.join(input_path, "*.JPEG"))
    val_videos = glob.glob(os.path.join(input_path, "*.JPEG"))
   
    vid_list={'train':videos,
              'val':val_videos}

    write_to_file(input_path,output_path,vid_list)
