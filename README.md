This project is based on https://github.com/Feynman27/pytorch-detect-to-track

Detect to Track and Track to Detect
    1) Purpose: 
Using a simple and effective way to solve the high accuracy detection and tracking.

    2) Architecture:
    1. A ConvNet had been set up for simultaneous detection and tracking, using a multi-task objective for frame-based object detection and across-frame track regression;
    2. Correlation features that represent object co-occurrences across time had been calculated to aid the ConvNet during tracking.
    3. The frame level detections based on the cross-frame tracklets had been linked to produce high accuracy detections at the video level.


    3) Experiments:
    I. Dataset sampling:
Original datasets:
There are 30 classes in 3862 training videos and the ImageNet object detection (DET) dataset (only using the 30 VID classes). Validation results on the 555 videos of ImageNet VID validation. From DET, at most 2k images per classes had been sampled. VID training set had been used 10 frames from each video.
However, this dataset does not include the person videos. To train this model, we need the continuous frame from videos. In this way, some improvements had been made.

    • Improvement:
The current datasets:
There are 101943 training images containing a total of 221823 objects. 
The class distribution in training dataset is present as the following:
'__background__': 0, 'bird': 6167, 'bus': 3305, 'car': 8883, 'dog': 7251, 'domestic_cat': 3853, 
'bicycle': 5102, 'motorcycle': 4521, 'watercraft': 4313, 'person': 178428.
    • The video class distribution in training dataset is present as the following: '__background__': 0, 
'bird': 3852, 'bus': 749, 'car': 5719, 'dog': 5036, 'domestic_cat': 
1701, 'bicycle': 3153, 'motorcycle': 1805, 'watercraft': 2069, 'person': 
69840.
    • The image class distribution in training dataset is present as the following: '__background__': 0, 
'bird': 2315, 'bus': 2556, 'car': 3164, 'dog': 2215, 'domestic_cat': 
2152, 'bicycle': 1949, 'motorcycle': 2716, 'watercraft': 2244, 'person': 
108588.

There are 89791 validation images containing a total of 183638 objects. 
The class distribution in validation dataset is present as the following: '__background__': 0, 'bird' : 9862, 'bus': 6313, 'car': 26690, 'dog': 20237, 'domestic_cat': 11629, 'bicycle': 8854, 'motorcycle': 1139, 'watercraft': 6089, 'person': 92825}

The additional video source is from: [1]
The fps (frame be second) of the videos in imageNet is around 25~30. And from the paper, we know that the author subsample 10 frames per video. It means the real fps is 1. 
In this way, we need to subsample AVA dataset with 1 fps. And subsample ActEV/VIRAT
With 30 fps.

    II. Dataset evaluation
    1. The matlab version validates the results from the frame and video level (2 frames at a time and 3 time at a time).
However, the pytorch version evaluates each frame from snippet of VAL.
    2. Since the original version lacks the evaluation loss, we cannot detect which epoch we can get the best epoch. 
        ◦ Improvement: To finish this, in each epoch, the evaluation loss had been calculated to compare and to obtain the best model at the suitable epoch. While the backbone is resnet18, the best model could be gained at eighth epoch.

    III. Training and testing
    1. Because of the imbalanced dataset, the accuracies of the different classes are not similar. Maybe it is needed to modify the codes in the minibatches. 
    • Improvement: Until now, the sampler method had been changed as “WeightedRandomSampler”. However, the learning rate can be increased as “1e-3”. It reminds me it may be caused by the additional person images.
And this new sampler method iterates the non-continuous index. But the images with non-continuous index have the different size. So, in this way, the batch size can only be set as 1.

    2. It is necessary to calculate the average training and validating loss, not only the step loss.
    • Improvement: Now the step loss had been renewed as average loss.

    3. Cannot change the “nw” (number of workers) from 0 to a positive number. Because we cannot re initialize CUDA in forked subprocess. To use CUDA with multiprocessing, we must use Python 3.4+ and the ‘spawn’ start method.
    4. There are some tips for training memory:[4]

    IV. Results
The following results are obtained from the trained model based on the basic pretrained backbone network (resnet50 and resnet18). 
    • Because of the hardware limitation [GPU], resnet101 could not be used.
    • If we changed the datasets, we cannot apply the pretrained base model-“rfcn_detect.pth”.
The models which got result (3) are stored in [7]


    • Comparing the training loss of the last iteration for each epoch in different models:
Without person => Dataset (1):

    • Involved person => Dataset (2):

    • Reason why not to use learning rate as 1e-3: 
In that case, the loss would become Nan. From my perspective, it might be caused by the imbalanced dataset numbers (More detailed, the number of person samples are far more than the other classes.) => loss will become nan
    • Since the iteration average training loss cannot cover the average epoch training loss. It is necessary to calculate the average training and validating loss, not only the step loss
    • The validation loss must be calculated to check if it is overfit.


