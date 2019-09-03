## Detect to Track

This project is based on https://github.com/Feynman27/pytorch-detect-to-track

There are some details:
Following the instruction of “https://github.com/Feynman27/pytorch-detect-to-track”, we could build the working environment. Besides those, some codes are added or modified for more fuctions.

Implement details:
-	The backbone networks are initialized in training process as you require. For example, if you need resnet18 as backbone network, please modified the codes, like “resnet (imdb_classes, 18, pretrained=True, class_agnostic=args.class_agnostic)”.
And then, download the target pretrained model in the specified folder “pretrained_model”. Load the pretrained parameters and use these to train our models. [The pretrained models are saved in this link: https://tuni-my.sharepoint.com/:f:/g/personal/yan_feng_tuni_fi/EjCH5QQDXfdLjsvjmISevgABlhY7VRz7rjPzYYjfpq1Gvg?e=e6L8vT]

-	To specify the class codes and names (in “lib\datasets\imagenet_detect.py”), “_classes” and “_classes_map” are modified as your requirements. The “classes” and “classes_map” (in “lib\datasets\vid_eval.py”) shall be modified as the previous. As well as the demo. “imagenet_vid_classes” in the demo codes shall be changed.

-	Since the previous version of “trainval.py” lacks the validation loss, the average loss for each epoch and the balanced sampling, the modified version “trainval_net_early_stopping.py” had been created for these functions.

-	In “other” folder, you can find the source codes for preparing the dataset. For example, the “Youtube.py” is responsible for downloading the videos on “YouTube”, converting them to frames, writing the annotation for the objects in these videos and listing the frame numbers in dataset folder of ImageNet.
