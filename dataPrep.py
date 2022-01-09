#How to install Yolov5

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


!git clone https://github.com/ultralytics/yolov5
!pip install -qr yolov5/requirements.txt
%cd yolov5

import torch
#from IPython.display import Image, clear_output
#from utils.google_utils import gdrive_download

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
