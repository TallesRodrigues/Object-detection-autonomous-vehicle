!pip install utils
from utils import *
import random
import numpy as np
import pandas as pd

import os
import shutil
import glob
from pathlib import Path
import pickle

import seaborn as sns
from matplotlib import pyplot as plt

import glob
from IPython.display import Image, display, clear_output

import torch

random.seed(42)

#How to install Yolov5

!git clone https://github.com/ultralytics/yolov5
!pip install -qr yolov5/requirements.txt
%cd yolov5

import torch
from IPython.display import Image, clear_output


clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Download the data
%%time
%cd "/content/drive/MyDrive/USP/"  #your path here
!curl -L "https://public.roboflow.com/ds/CtAm2NjHfb?key=gjWvPQ9TcM" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip  #your url there

#Folders
trainPath      = '/content/drive/MyDrive/USP/export/train/'
trainImagePath = '/content/drive/MyDrive/USP/export/train/images/'
trainLabelPath = '/content/drive/MyDrive/USP/export/train/labels/'
testPath       = '/content/drive/MyDrive/USP/export/test/'
testImagePath  = '/content/drive/MyDrive/USP/export/test/images/'
testLabelPath  = '/content/drive/MyDrive/USP/export/test/labels/'
validPath      = '/content/drive/MyDrive/USP/export/valid/'
validImagePath = '/content/drive/MyDrive/USP/export/valid/images/'
validLabelPath = '/content/drive/MyDrive/USP/export/valid/labels/'
dataExplorationPath = "/content/drive/MyDrive/USP/dataExploration/"

Paths_dir = [trainPath,trainImagePath,trainLabelPath,
             testPath,testImagePath,testLabelPath,
             validPath,validImagePath,validLabelPath]

#Creating folders
for dir in Paths_dir:
  os.mkdir(dir)

  
#Exploring the dataset
labelPath = '/content/drive/MyDrive/USP/export/labels/'
imagePath = '/content/drive/MyDrive/USP/export/images/'

allLabels = []
notLabels = []

for filename in os.listdir("/content/drive/MyDrive/USP/export/labels"):
  mypath = Path(labelPath+filename)
  if filename.endswith(".txt"):
    if (not mypath.stat().st_size == 0):
      allLabels.append(filename)
    else:
      notLabels.append(filename)

print(f"All labels files = {len(allLabels)}")
print(f"Files without label = {len(notLabels)}")

# Creating a dataframe with the data from the files
%%time
row_list = []

for i,f in enumerate(allLabels):
  with open(labelPath+f,mode='r') as file_explorer:
    lines = file_explorer.readlines()
    print(i+1,"/",len(allLabels))
    for line in lines:
      dic = {}
      obj,xi,yi,xu,yu = line.split(' ')
      dic['Object'] = obj
      dic['xi']     = xi
      dic['yi']     = yi
      dic['xu']     = xu
      dic['yu']     = yu.strip('\n')
      dic['image_path'] = imagePath + file_explorer.name.split('/')[-1].replace('.txt','.jpg')
      dic['file_path']  = file_explorer.name
      row_list.append(dic)

#Saving the dataframe to a file for future exploration
data = pd.DataFrame(row_list)
data.to_pickle(dataExplorationPath+"compiled_label_data.pkl")

# Droping duplicates on the dataframe
dt = data.drop_duplicates(subset=['Object','xi','yi','xu','yu'])

# Ensuring data type
dt['Object']= dt['Object'].astype('int32')

# Checking the number of labels per class
y = pd.DataFrame(dt['Object'].value_counts())
names= ['biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft', 'trafficLight-Red', 'trafficLight-RedLeft', 'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck']
ind_values = [names[x] for x in y.index]
y.index = ind_values
print(y)

# Plotting the class distribution
sns.set(rc={'figure.figsize':(11.7,8.27)})
ax = sns.barplot(x=y.index,y='Object',data=y)
ax.tick_params(axis='x', rotation=90)
ax.set_title("Distribuição das classes",size=20)

# Saving the cleaned dataframe
dt.to_pickle('/content/drive/MyDrive/USP/dataExploration/'+"dataFrame_Clean.pkl")


# Spliting the data between training and test
Images_train, Images_test, label_train, label_test = train_test_split(dt['image_path'],dt['file_path'], test_size=0.33)
print(f"Train and test  = {len(Images_train), len(Images_test)}")

# Moving images and labels to the desired folders
def move_images_labels(Images_train,Images_test,Images_validation,
                       Train_labels,Test_labels,Valid_labels):
   
  #moving images

  destination = "/content/drive/MyDrive/USP/export/train/images/"
  if Images_train.notnull:
    for train in Images_train:
      shutil.move(train, destination+train.split('/')[-1])
    print('Train Images copied')  

  destination = "/content/drive/MyDrive/USP/export/test/images/"
  if Images_test.notnull:
    for test in Images_test:
      shutil.move(test , destination+test.split('/')[-1])
    print('Test Images copied')  

  destination = "/content/drive/MyDrive/USP/export/valid/images/"
  if len(Images_validation)>0:
    for valid in Images_validation:
      shutil.move(valid , destination+valid.split('/')[-1])
    print('Validation Images copied')  

  #moving labels
  
  destination = "/content/drive/MyDrive/USP/export/train/labels/"
  if Train_labels.notnull:
    for train in Train_labels:
      shutil.move(train , destination+train.split('/')[-1])
    print('Train labels copied')  

  destination = "/content/drive/MyDrive/USP/export/test/labels/"
  if Test_labels.notnull:
    for test in Test_labels:
      shutil.move(test , destination+test.split('/')[-1])
    print('Test labels copied')  

  destination = "/content/drive/MyDrive/USP/export/valid/labels/"
  if len(Valid_labels)>0:
    for valid in Valid_labels:
      shutil.move(valid, destination+valid.split('/')[-1])
    print('Validation labels copied')  
  
  pass

move_images_labels(Images_train,'/content/drive/MyDrive/USP/export/train/images/',label_train,'/content/drive/MyDrive/USP/export/train/labels/')
move_images_labels(Images_test,'/content/drive/MyDrive/USP/export/test/images/',label_test,'/content/drive/MyDrive/USP/export/test/labels/')
move_images_labels(list(Images_validation.keys()),'/content/drive/MyDrive/USP/export/valid/images/',list(Images_validation.values()),'/content/drive/MyDrive/USP/export/valid/labels/')


from sklearn.model_selection import train_test_split

