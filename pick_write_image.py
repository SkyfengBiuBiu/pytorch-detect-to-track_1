#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:47:53 2019

@author: fengy
"""
from pytube import YouTube
import os
import csv
import glob
import numpy as np
import time
import cv2
import xml.etree.ElementTree as ET
from PIL import Image 
import sys, os, glob, shutil


def writeAnnotation(image_name, person_list, output_file,resolution_pair):
    
    try:
        os.mkdir(output_file)
        print("Output folder :\n%s was created. " % output_file)
    except OSError:
        #print('There is an error of creating the specified folder')
        exit()
    
    print(person_list)

    annotation = ET.Element('annotation')  
    folder = ET.SubElement(annotation, 'folder')  
    filename = ET.SubElement(annotation, 'filename')
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')  
    size = ET.SubElement(annotation, 'size')  

    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')  
  

    for bbox in person_list:
        object_ = ET.SubElement(annotation, 'object')
        name = ET.SubElement(object_, 'name')  
        bndbox = ET.SubElement(object_, 'bndbox') 
    
        xmax = ET.SubElement(bndbox, 'xmax')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymax = ET.SubElement(bndbox, 'ymax')  
        ymin = ET.SubElement(bndbox, 'ymin') 
        name.text = 'n00007846'  
    
        xmax.text = str(int(float(bbox[1])*resolution_pair[0]))
        xmin.text = str(int(float(bbox[0])*resolution_pair[0]))
        ymax.text = str(int(float(bbox[3])*resolution_pair[1]))
        ymin.text = str(int(float(bbox[2])*resolution_pair[1]))
        
    #To do
    folder.text = 'FIGURE_EIGHT/'+'n00007846'  
    filename.text = image_name
    database.text = 'FIGURE_EIGHT'  
    width.text = str(resolution_pair[0])
    height.text = str(resolution_pair[1])  
    output_file_name=os.path.join(output_file,"{}.xml".format(image_name))
    tree = ET.ElementTree(annotation)
    tree.write(output_file_name)





def readCSV():
    path="/home/fengy/Downloads/figure_eight"
    class_csv=os.path.join(path,"class-descriptions-boxable.csv")
    bbox_csv=os.path.join(path,"train-annotations-bbox.csv")
    
    label_names=[]
    image_box_dict={}
    person_interest=['Man']
    person_box_list=[]
    
    with open(class_csv, mode='r') as csv_file:
        csv_reader=csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[1] in person_interest:
                label_names.append(row[0])
    label_names=np.unique(label_names).tolist()

    with open(bbox_csv, mode='r') as csv_file:
        csv_reader=csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[2] in label_names:
                if row[0] in image_box_dict:
                    person_box_list=image_box_dict[row[0]]
                    person_box_list.append(row[4:8])
                    image_box_dict.update({row[0]:person_box_list})
                else:
                    image_box_dict.update({row[0]:[row[4:8]]})
            
    return image_box_dict

#%%
if __name__=="__main__":

    segments=['train','val']
    original_path="/home/fengy/Downloads/train_08"
    image_path='/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/Data/DET/'
    save_anno_path='/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/Annotations/DET/'
    
    image_box_dict=readCSV()
    num_image_box=len(image_box_dict)
    keys=image_box_dict.keys()
    train_dict={k:image_box_dict[k] for k in keys[:int(np.ceil(num_image_box/8*7))]}
    val_dict={k:image_box_dict[k] for k in keys[int(np.ceil(num_image_box/8*7)):]}
    seg_dict={'train':train_dict,
              'val':val_dict}
    
    for segment in segments:
        image_path_s=os.path.join(image_path,segment,'FIGURE_EIGHT')
        anno_output=os.path.join(save_anno_path,segment,'FIGURE_EIGHT')
        if not os.path.exists(image_path_s):
            os.makedirs(image_path_s)
        if not os.path.exists(anno_output):
            os.makedirs(anno_output)
            
        image_path_s=os.path.join(image_path_s,'n00007846')
        anno_output=os.path.join(anno_output,'n00007846')
        if not os.path.exists(image_path_s):
            os.makedirs(image_path_s)
        if not os.path.exists(anno_output):
            os.makedirs(anno_output)
               
        
        segment_dict=seg_dict[segment]
        images_input_path=glob.glob(os.path.join(original_path,"*.jpg"))
        for image in images_input_path:
            image_name=os.path.basename(image)
            image_id=os.path.splitext(image_name)[0]
            if image_id in segment_dict.keys():
                im=Image.open(image)
                width, height = im.size
                im.save(os.path.join(image_path_s,(image_id+".JPEG")))
                output_file=anno_output
                writeAnnotation(image_id, segment_dict[image_id], output_file,(int(width), int(height)))
                print("successfully!"+image_name)
        