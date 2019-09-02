from pytube import YouTube
import os
import csv
import glob
import numpy as np
import time
import cv2
import xml.etree.ElementTree as ET

def writeAnnotation(video_name,frame_id, person_dict, output_file,resolution_pair):
    
    try:
        os.mkdir(output_file)
        print("Output folder :\n%s was created. " % output_file)
    except OSError:
        #print('There is an error of creating the specified folder')
        exit()
    
    print(person_dict)

    annotation = ET.Element('annotation')  
    folder = ET.SubElement(annotation, 'folder')  
    filename = ET.SubElement(annotation, 'filename')
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')  
    size = ET.SubElement(annotation, 'size')  

    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')  
  

    for person_id, bbox in person_dict.items():
        object_ = ET.SubElement(annotation, 'object')
        trackid = ET.SubElement(object_, 'trackid')
        name = ET.SubElement(object_, 'name')  
        bndbox = ET.SubElement(object_, 'bndbox') 
    
        xmax = ET.SubElement(bndbox, 'xmax')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymax = ET.SubElement(bndbox, 'ymax')  
        ymin = ET.SubElement(bndbox, 'ymin') 
    
        occluded = ET.SubElement(object_, 'occluded')  
        generated = ET.SubElement(object_, 'generated') 
    
        trackid.text = str(person_id)
        name.text = 'n00007846'  
    
        xmax.text = str(int(float(bbox[2])*resolution_pair[0]))
        xmin.text = str(int(float(bbox[0])*resolution_pair[0]))
        ymax.text = str(int(float(bbox[3])*resolution_pair[1]))
        ymin.text = str(int(float(bbox[1])*resolution_pair[1]))
        occluded.text = '1'
        generated.text = '1' 
        
    #To do
    folder.text = 'AVA/'+video_name  
    filename.text = "%#06d" % (frame_id)
    database.text = 'AVA'  
    width.text = str(resolution_pair[0])
    height.text = str(resolution_pair[1])  
    output_file_name=output_file + "/%#06d.xml" % (frame_id)
    tree = ET.ElementTree(annotation)
    tree.write(output_file_name)

def downloadYouTube(video_id,segment,count):
    
     width=0.0
     height=0.0
     default_link="https://www.youtube.com/watch?v="
     try:
         yt = YouTube(default_link+video_id)
     except:
         print("Connection error!")
         return False, {},count,{}
     
     yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
     path=os.path.join('/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/Data/VID/snippets/',segment,'AVA')
     if not os.path.exists(path):
        os.makedirs(path)
     try:
         name="AVA_{}_{}".format(segment,str(count))
         yt.download(path, filename=name)
         
         vcap = cv2.VideoCapture(os.path.join(path,name+".mp4")) # 0=camera

         if vcap.isOpened(): 
            width = vcap.get(3)  # float
            height = vcap.get(4) # float
         return True, {video_id:name},(count+1),{name:(int(width),int(height))}
     except:
         print("Forbidden!")
         return False, {},count,{}

   
def readCSV(segment):
    path="/home/fengy/Downloads/ava_v2.2"
    csv_paths=glob.glob(os.path.join(path,'*.csv'))
    pick_file=""
    video_ids=[]
    person_box_dict={}
    frame_box={}
    for csv_path in csv_paths:
        if segment in csv_path:
            pick_file=csv_path
    with open(pick_file, mode='r') as csv_file:
        csv_reader=csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            video_ids.append(row[0])
    video_ids=np.unique(video_ids).tolist()
# =============================================================================
#     video_num=min(len(video_ids),1)
#     video_ids=video_ids[:video_num]
#     
# =============================================================================
    with open(pick_file, mode='r') as csv_file:
        csv_reader=csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] in video_ids:
                new_key = "%d_%d" % (int(row[1]), int(row[-1]))
                if row[0] in person_box_dict:
                    frame_box=person_box_dict[row[0]]
                    frame_box.update({new_key:row[2:6]})
                    person_box_dict.update({row[0]:frame_box})
                else:
                    person_box_dict.update({row[0]:{new_key:row[2:6]}})
                
                
            
    return video_ids, person_box_dict

def video_to_frames(input_loc, output_loc,frame_unique_sort_ids):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
        print("Output folder :\n%s was created. " % output_loc)
    except OSError:
        print('There is an error of creating the specified folder')
        pass
    # Log the time
    time_start = time.time()
    count = 0
    sec = 0
    frameRate = 1.0
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    
    success,image = cap.read()
 #//it will capture image in each 0.5 second
    while (success and count<=frame_unique_sort_ids[-1]):
    
        if count in frame_unique_sort_ids:
            cv2.imwrite(output_loc + "/%#06d.JPEG" % (count), image)   # save frame as JPEG file      
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        success,image = cap.read()
        print('Read a new frame: ', count)
        count += 1
        sec = sec + frameRate
        sec = int(sec)
    time_end = time.time()
    print ("Done extracting frames.\n%d frames extracted" % count)
    print ("It took %d seconds forconversion." % (time_end-time_start))



#%%    
if __name__=="__main__":
    
    segments=['train','val']
    destination_directory='/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/Data/VID/snippets/'
    image_path='/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/Data/VID/'
    save_anno_path='/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/Annotations/VID/'
    
    segment_count={
            'train':40,
            'val':10}
    
    for segment in segments:
        video_path_s=os.path.join(destination_directory,segment,'AVA')
        image_path_s=os.path.join(image_path,segment,'AVA')
        anno_output=os.path.join(save_anno_path,segment,'AVA')
        if not os.path.exists(image_path_s):
            os.makedirs(image_path_s)
        if not os.path.exists(anno_output):
            os.makedirs(anno_output)
        
        video_ids, person_box_dict=readCSV(segment)
        select_downloaded_video={}
        person_box_dict1={}
        video_resolution={}
        idx=0
        for _, video in enumerate(video_ids):
            flag,videoMapping,idx,select_resolution=downloadYouTube(video,segment,idx)
            if flag:
                print("Downloading the video {}".format(str(idx)))
                select_downloaded_video.update(videoMapping)
                video_resolution.update(select_resolution)
            if idx>=segment_count[segment]:
                break
        

        for person_b in person_box_dict:
            if person_b in select_downloaded_video:
                person_box_dict1.update({select_downloaded_video[person_b]:person_box_dict[person_b]})
        person_box_dict=person_box_dict1
        
        
        for video_b in person_box_dict:
            print("Annotate the video {}".format(video_b))
            anno_output_file=os.path.join(anno_output,video_b)
            resolution_pair=video_resolution[video_b]
            person_boxes=person_box_dict[video_b]
            frame_person=person_boxes.keys()
            frame_ids=[]
            person_ids=[]
            for key in frame_person:
                frame_id=key.split('_')[0]
                person_id=key.split('_')[1]
                frame_ids.append(frame_id)
                person_ids.append(person_id)
    
            frame_unique_ids=np.unique(frame_ids).tolist()
            results = map(int, frame_unique_ids)
            frame_unique_sort_ids=sorted(results)
    
            person_unique_ids=np.unique(person_ids).tolist()
            person_results = map(int, person_unique_ids)
            person_unique_sort_ids=sorted(person_results)
                

            for frame in frame_unique_sort_ids:
                person_dict={}
                for person in person_unique_sort_ids:
                    new_key = "%d_%d" % (frame, person)
                    if new_key in frame_person:
                        person_dict.update({person: person_boxes[new_key]})
                if bool(person_dict):
                    writeAnnotation(video_b,frame, person_dict,anno_output_file,resolution_pair)

            
            print("Converting the video {} to frame".format(video_b))
            output_loc=os.path.join(image_path_s,video_b)
            input_loc=os.path.join(video_path_s,("{}.mp4".format(video_b)))
            video_to_frames(input_loc, output_loc,frame_unique_sort_ids)   
            #video_to_frames1(input_loc, output_loc)  
