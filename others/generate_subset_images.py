import os
import pdb
import numpy as np
import scipy.sparse
#import cPickle
import uuid
import torch

class generate_subset_images(object):
    ##
    # @brief To initialize the dataset reader
    #
    # @param image_set One of [train,val,trainval,test]
    # @param devkit_path Full path to the directory where the data created by
    # imagenet_datasets.py is stored
    #
    def __init__(self, image_set, devkit_path, det_or_vid):

        self._det_vid = det_or_vid 
        self._root_path = devkit_path
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = devkit_path # Currently its same as the devkit_path

        self._classes = ('__background__',  # always index 0
                         'bird', 'bus', 'car', 
                         'dog', 'domestic_cat', 'bicycle',                         
                          'motorcycle', 'watercraft')
         
 
        self._classes_map = ('__background__',  # always index 0
                         'n01503061', 'n02924116', 'n02958343', 
                         'n02084071', 'n02121808',   'n02834778',                    
                         'n03790512', 'n04530566')
         
# =============================================================================
#         self._classes = ('__background__',  # always index 0
#                          'airplane', 'antelope', 'bear', 'bicycle',
#                          'bird', 'bus', 'car', 'cattle',
#                          'dog', 'domestic_cat', 'elephant', 'fox',
#                          'giant_panda', 'hamster', 'horse', 'lion',
#                          'lizard', 'monkey', 'motorcycle', 'rabbit',
#                          'red_panda', 'sheep', 'snake', 'squirrel',
#                          'tiger', 'train', 'turtle', 'watercraft',
#                          'whale', 'zebra')
#          
#  
#         self._classes_map = ('__background__',  # always index 0
#                          'n02691156', 'n02419796', 'n02131653', 'n02834778',
#                          'n01503061', 'n02924116', 'n02958343', 'n02402425',
#                          'n02084071', 'n02121808', 'n02503517', 'n02118333',
#                          'n02510455', 'n02342885', 'n02374451', 'n02129165',
#                          'n01674464', 'n02484322', 'n03790512', 'n02324045',
#                          'n02509815', 'n02411705', 'n01726692', 'n02355227',
#                          'n02129604', 'n04468005', 'n01662784', 'n04530566',
#                          'n02062744', 'n02391049')
# =============================================================================
        
        self.num_classes=len(self._classes)
        print("Number of classes: {}".format(self.num_classes))
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        self._image_ext = '.JPEG'
        
    def _load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self._data_path, 'ImageSets',
                self._det_vid, self._image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            lines = [x.strip().split(' ') for x in f.readlines()]
        if len(lines[0]) == 2:
            self._image_index = [x[0] for x in lines]
        else:
            self._image_index = ['%s' % x[0] for x in lines]
        

            
    def _delete_lines(self,valid_index):
        
        image_set_index_file = os.path.join(self._data_path, 'ImageSets',
                self._det_vid, self._image_set + '.txt')
        
        with open(image_set_index_file, "r") as f:
            lines = f.readlines()
        with open(image_set_index_file, "w") as f:
            for idx, line in enumerate(lines):
                if valid_index[idx]:
                    f.write(line)
            

        
    def _load_vid_annotation(self, idx, index):


        import xml.etree.ElementTree as ET
        
        #index = index.replace("/", "\\")
        flag=False
        
        if self._det_vid == 'DET':
            filename = os.path.join(self._data_path, 'Annotations', 'DET', self._image_set, index + '.xml')
        else:
            filename = os.path.join(self._data_path, 'Annotations', 'VID', self._image_set, index + '.xml')

        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        
        class_to_index = dict(zip(self._classes_map, xrange(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            #if class_to_index.has_key(obj.find('name').text):
            #print(obj.find(str('name')))
            if  (obj.find('name').text) in class_to_index:
                flag=True
                break

        return flag
    
    def _load_valid_index(self, valid_index, image_index_pre,flag):
        
        sub_chunk=[]
        new_chunk=[]
        
        if flag:
            for i in range(len(valid_index)):
                if i==0:
                    sub_chunk.append(valid_index[i])
                elif image_index_pre[i]==image_index_pre[i-1]:
                    sub_chunk.append(valid_index[i])
                else:
                    if True in sub_chunk:
                        sub_chunk=[True]*len(sub_chunk)
                    new_chunk.append(sub_chunk)                
                    sub_chunk=[]
                    sub_chunk.append(valid_index[i])
           

            new_chunk.append(sub_chunk)
     
            concat_list = [j for i in new_chunk for j in i] 
            return concat_list
        else:
            return valid_index

    
if __name__ == '__main__':
    
    dataset={}
    devkit_path = os.path.join('data', 'ILSVRC')
    
    
    for split in ['VID']:
        for select in ['val']:
            new_class=generate_subset_images(select,devkit_path,split)
            new_class._load_image_set_index()

            valid_index = np.zeros((len(new_class._image_index)), dtype=np.bool)
            image_index_pre = ['']*len(new_class._image_index)
            image_index = new_class._image_index
    
            for idx, index in enumerate(image_index):
                valid_index[idx]= new_class._load_vid_annotation(idx,index)
                pre = image_index[idx].split('/')
                image_index_pre[idx]= pre[0]+'/'+pre[1]
                
            VID_flag=(split=='VID')
            val_flag=(select=='val')
   
            #valid_index=new_class._load_valid_index(valid_index, image_index_pre,(VID_flag and ( not val_flag)))        
    
            new_class._delete_lines(valid_index)
    


    #print(concat_list)
 
 

# =============================================================================
#     valid_index[0]= new_class._load_vid_annotation(0,new_class._image_index[0])
#     print(valid_index)
#     new_class._delete_lines(valid_index)
# =============================================================================
 
# =============================================================================
#     for select in ['train', 'val', 'test']:
#     name = 'imagenet_vid_{}'.format(split)
#     devkit_path = os.path.join('data', 'ILSVRC')
#     data_path = os.path.join('data', 'ILSVRC')
#     __sets[name] = (lambda split=split, devkit_path=devkit_path: imagenet_detect(split,devkit_path, 'VID'))
#     
# =============================================================================
