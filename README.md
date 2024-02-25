# PyDet
Module with supplier functions oriented around computer vision

To use this module, download this repository and add it to the folder you need PyDet for. The necessary files/folders are:

- build
- models
- pybind11
- yolov5-master
- CMakeLists.txt
- Makefile
- cmake_install.cmake
- filters.cpp
- pydet.py

To download, run ```git clone https://github.com/v2pir/PyDet.git```

After adding this to your folder, to use PyDet, simply type:

```import pydet as pd```

You can then use the functions defined.

# Documentation

# Detect
- detectBlob(): detects blob in image
- detectTumor(): Uses a pre-trained YOLOv5 model to detect brain tumors in MRI scans
- detectPet(): Uses a pre-trained YOLOv5 model to detect animals in images
- detectHuman(): Uses a pre-trained YOLOv5 model to detect humans in images
- detectText(): Uses OCR to detect text in images

# Colors
- findSpecificColor(): given an HSV range and an image parameter (string to its relative path), it will detect any contour within that HSV range
- findColor(): pre-defined HSV ranges for common colors including: ```ORANGE, RED, GREEN, BLUE, PINK, YELLOW```

# Random set of functions
- splitVideo(): splits a video into frames
- getOrientation(): returns the yaw, pitch, roll, and camera position, of a given object in the image (3d array of object dimensions + 2d array of coordinates of the object in the image is required)
- LinRegLossFunc(): returns the losses and weights of a linear regression model

# Filters
- saturate(): will saturate an image by a given intensity (the function defined in ```filters.cpp``` which is why pybind11/CMake is needed)
