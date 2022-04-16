#Computer Vision 

#!git clone https://github.com/ultralytics/yolov5   #clone yolov5 git repo before implemnting the file

#%cd yolov5
#%pip install -qr requirements.txt

#importing pytorch.
import torch
from yolov5 import utils
display = utils.notebook_init().

#Unziping the training data.
#!unzip -q ../train_data.zip -d../ 

#Train YOLOv5s on COCO128 for 3 epochs
!python train.py --img 640 --batch 16 --epochs 39 --data coco128.yaml --weights yolov5s.pt --cache

#Predicting
!python detect.py --weights runs/train/exp10/weights/best.pt --img 640 --conf 0.25 --source 1e799e950e349c438017d88fe92a44c2--super-star-mens-hairstyle.jpg
