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


'''
Base class for Parsing all the datasets
'''



class ImagenetVID():
    def __init__(self, data_dir, dataset_path):
        # Calling the base class constructor first

        # Store the dataset path
        self.dataset_path = dataset_path
        self.unique_classes = []

        # Read the xml annotation files

        # Get all the images that are present in the dataset
        self.im_list = []
        self.img_to_annot_map = {}
        #self.vid_list = {'train':{}, 'val':{}}
        self.vid_list = {'val':{}}
        self.get_vid_list()

    def write_to_file(self):
        for segment in self.vid_list:
            fname = os.path.join(output_path, segment) + ".txt"

#            if os.path.exists(fname):
#                 os.remove(fname)

            print("Writing to {}".format(fname))
            for video in self.vid_list[segment]:
                if len(self.vid_list[segment][video])==0: continue
                last_frame=int(self.vid_list[segment][video][-1].split('.')[0])
                for frame in self.vid_list[segment][video]:
                    frame_number = int(frame.split('.')[0])
                    with open(fname,"a+") as f:
                        #f.write(os.path.join(video,frame.split('.')[0]) + " 1" + " " + str(frame_number) + " " + str(last_frame) + "\n")
                        f.write(os.path.join(video,frame.split('.')[0]) + " " + str(frame_number) + "\n")
    def merge_train_val(self):
        raise NotImplementedError

    def get_vid_list(self):
        np.random.seed(1)
        # Iterate over train/val/test
        for segment in os.listdir(self.dataset_path):
            if segment not in self.vid_list: continue
            # Build list of video snippets for each segment
            seg_path = os.path.join(self.dataset_path, segment)
            n_frames = 0
            for i,vid in enumerate(os.walk(seg_path)):
                if i==0 or len(vid[2])==0:
                    print(vid[0])
                    continue
                frame_list = sorted(vid[2])
                if frames_per_video != -1:
                    #frame_list = frame_list[0::int(np.ceil(len(frame_list) / float(frames_per_video)))]
                    frame_list = frame_list[:]


                n_frames += len(frame_list)
                if os.path.basename(vid[0]) not in self.vid_list[segment]:
                    self.vid_list[segment][os.path.basename(vid[0])]=[]
                
                path=vid[0]

                 #For the whole dataset
# =============================================================================
#                 if path.split("/")[-2] == segment:
#                      self.vid_list[segment][os.path.basename(vid[0])] = frame_list
#                 elif path.split("/")[-2] == "VIRAT-V1":
#                      continue
#                 else:
#                      self.vid_list[segment][os.path.join(path.split("/")[-2],path.split("/")[-1])] = frame_list
# 
# =============================================================================
                    
                #Only for people
                if path.split("/")[-2] == "VIRAT-V1" :
                    self.vid_list[segment][os.path.join(path.split("/")[-2],path.split("/")[-1])] = frame_list
                else:
                    continue

            print("Total frames in {}:{}".format(segment,n_frames))




# To get the name of class from string
def str_to_classes(str):
    curr_class = None
    try:
        curr_class = getattr(sys.modules[__name__], str)
    except:
        print "Dataset class is not implemented"
    return curr_class


if __name__ == "__main__":

    output_path='./data/ILSVRC/ImageSets/VID'
    input_path='./data/ILSVRC/Annotations/VID'
    dataset='ImagenetVID'
    frames_per_video=100


    data_path = output_path
    datasets = [dataset]
    dataset_paths = [input_path]
    # Process all the datasets
    for dataset, dataset_path in zip(datasets, dataset_paths):
        curr_dataset = str_to_classes(dataset)(data_path, dataset_path)
        curr_dataset.write_to_file()
