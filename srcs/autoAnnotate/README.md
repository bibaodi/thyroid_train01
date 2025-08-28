# Auto Annotate the images by existed CNN models
- author: eton
- Version: V0.0.1 d250828

## Prompts
1. first version prompts
> please help me create a python application which use a CNN model generate annotation from an input image , result in lablme json format which define in LabelmeJson.py.  the requirements is: 1. support input cli parameters are {1. input model file ; 2. input model type(segmentation or detection; 3. image files folder which will input to model ; 4. output folder which will contains the result json file.} ; 2. the process include several steps {1. check all input arguments and do illegal check; 2. use exist logger , record all necessary information to log file; 3. iterate all images and use model to do predict; 4. convert the predict result to labelme json format, add the image file name to imagePath; 5. keep the json file relative path same with image relative path, just the root prefix from input argument} ; 3. now the mode is from Yolo-11, you can test it use '/tmp/model_segmentThyGland_v02.250821.pt'; 4. please follow python application best practice; 5. add necessary comments and file header; 6. make application run from main() function, add unit test function if it is needed.

# END./
