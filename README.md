# Object-detection-autonomous-vehicle
YoloV5 Applied to object detection

How to install Yolov5

!git clone https://github.com/ultralytics/yolov5
!pip install -qr yolov5/requirements.txt
%cd yolov5

import torch
#from IPython.display import Image, clear_output
#from utils.google_utils import gdrive_download

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
