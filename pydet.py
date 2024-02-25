 # Please see README.md for explanation of options.
# https://github.com/v2pir

import torch
import cv2
import os
import site

YOLOV5_DIR = site.getsitepackages()[0] + "/yolov5-master"

class Detect:

    def __init__(self, img_path:str):
        self.img_path = img_path

    def detectBlob(self):

        image = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)

        detector = cv2.SimpleBlobDetector_create()
        
        # Detect blobs
        keypoints = detector.detect(image)

        im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imwrite("blob_keypoints.jpg", im_with_keypoints)

        return keypoints

    def detectTumor(self):

        tumor_model = torch.hub.load(YOLOV5_DIR, 'custom', path=site.getsitepackages()[0]+"/models/tumor.pt", source='local')

        #read image
        img = cv2.imread(self.img_path)

        #convert image to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #run model on image and save image to runs/detect/exp
        results = tumor_model(img)
        res = results.xyxy[0]

        data = []

        for predictions in res:
            x1, y1, x2, y2, confidence, class_number = map(float, predictions)
            if class_number == 0 and confidence > 0.3:
                data.append([x1,y1,x2,y2,confidence,class_number])

                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 5)

                cv2.putText(img, str(confidence), (20,50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (150, 150, 0), thickness=3)

        cv2.imwrite('tumor.jpg', img)

        return data

    def detectPet(self):

        pet_model = torch.hub.load(YOLOV5_DIR, 'custom', path=site.getsitepackages()[0]+"/models/pet.pt", source='local')

        #read image
        img = cv2.imread(self.img_path)

        #convert image to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #run model on image and save image to runs/detect/exp
        results = pet_model(img)
        res = results.xyxy[0]

        data = []

        for predictions in res:
            x1, y1, x2, y2, confidence, class_number = map(float, predictions)
            if class_number == 0 and confidence > 0.3:
                data.append([x1,y1,x2,y2,confidence,class_number])

                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 5)

                cv2.putText(img, str(confidence), (20,50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (150, 150, 0), thickness=3)

        cv2.imwrite('dog.jpg', img)

        return data
    
    def detectHuman(self):

        human_model = torch.hub.load(YOLOV5_DIR, 'custom', path=site.getsitepackages()[0]+"/models/human.pt", source='local')

        #read image
        img = cv2.imread(self.img_path)

        #convert image to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #run model on image and save image to runs/detect/exp
        results = human_model(img)
        res = results.xyxy[0]

        data = []

        for predictions in res:
            x1, y1, x2, y2, confidence, class_number = map(float, predictions)
            if class_number == 0 and confidence > 0.3:
                data.append([x1,y1,x2,y2,confidence,class_number])

                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 5)

                cv2.putText(img, str(confidence), (20,50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (150, 150, 0), thickness=3)

        cv2.imwrite('human.jpg', img)

        return data
    
    def detectText(self):
        img = cv2.imread(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3,3), 0)
        mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Morph open to remove noise and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        res = 255 - opening
       
        string = (pytesseract.image_to_string(res, lang="eng", config='--psm 6'))
        string = string.replace("\n", " ")
        return [string]

import os
import shutil
import pytesseract
import numpy as np
import math
import matplotlib.pyplot as plt


class Colors:

    def __init__(self, img_name:str):
        self.img_name = img_name

    def findSpecificColor(self, hsv_low:tuple, hsv_high:tuple, imgout:bool, coordsout:bool):
        #read image
        img = cv2.imread(self.img_name)

        #convert image to hsv
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img,hsv_low, hsv_high)

        if imgout:
            cv2.imwrite("findColor.jpg", mask)


        if coordsout:
            contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image
            coords = []

            #find position of pathmarker color
            if len(contour) != 0:
                for contour1 in contour:
                    if cv2.contourArea(contour1) > 500:
                        x, y, w, h = cv2.boundingRect(contour1)
                        coords.append([(x,y), (x, y+h), (x+w, y), (x+w, y+h)])
            return coords
        
        else:
            return None

    def findColor(self, color:str, imgout:bool, coordsout:bool):
        #read image
        img = cv2.imread(self.img_name)

        #convert image to hsv
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if color == "ORANGE":
            mask = cv2.inRange(img,(3, 100, 20), (15, 255, 255))
        elif color == "RED":
            mask = cv2.inRange(img,(0, 165, 20), (179,255,255))
        elif color == "GREEN":
            mask = cv2.inRange(img,(40, 40, 40), (70,255,255))
        elif color == "BLUE":
            mask = cv2.inRange(img,(100, 60, 0), (140,255,255))
        elif color == "PINK":
            mask = cv2.inRange(img,(120, 10, 50), (180,255,255))
        elif color == "YELLOW":
            mask = cv2.inRange(img,(15, 100, 100), (30,255,255))
        else:
            print("\033[1;31;40m" + "Error: Not a color. Pick, ORANGE, RED, GREEN, BLUE, PINK, or YELLOW.\nOr use findSpecific()")
            return False

        if imgout:
            cv2.imwrite("maskedColor.jpg", mask)

        if coordsout:
            #get contour
            contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image
            coords = []

            #find position of pathmarker color
            if len(contour) != 0:
                for contour1 in contour:
                    if cv2.contourArea(contour1) > 500:
                        x, y, w, h = cv2.boundingRect(contour1)
                        coords.append([(x,y), (x, y+h), (x+w, y), (x+w, y+h)])

            return coords
        else:
            return None


    
#splits a video at around 32fps into frames and puts it in a folder called "image_frames"
def splitVideo(video:str):

    #open video
    cap = cv2.VideoCapture(video)
    img_num = 0

    #make directory
    path = os.getcwd()
    path = os.path.join(path, "image_frames")
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)

    while cap.isOpened():
        try:
            #get frame
            _, frame = cap.read()

            if frame is None:
                break

            #add frame to directory
            cv2.imwrite(path + "/IMG" + str(img_num) + ".jpg", frame)

            img_num += 1
        except:
            break


def getOrientation(array2d:list, array3d:list, image:str, dist_coeffs = np.zeros((5,1))):

    '''
    to calculate your own dist_coeffs, use cv2.calibrateCamera(*your params*)[2], which returns an nx1 matrix that you can use

    '''

    image = cv2.imread(image)
    n = len([array2d])
    dim = image.shape

    arr_3d = np.array(array3d,

                        dtype= 'float32')
    
    points_3D = np.stack([arr_3d]*n)  # adds n copies of array1_3D (3d coordinates of the buoy) into one array
    points_2D = np.array(
        
                        [array2d],

                        dtype= 'float32')

    center = (dim[1]/2, dim[0]/2)

    mtx_cam = np.array(
                            [[dim[1], 0, center[0]],
                            [0, dim[1], center[1]],
                            [0, 0, 1]], dtype = 'float32'
                            )
    mtx_cam_2 = np.asarray(mtx_cam,np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(points_3D, points_2D, mtx_cam_2, dist_coeffs, flags= 0)

    np_rodrigues = np.asarray(rotation_vector[:,:],np.float64)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)

    yaw = -math.degrees(math.asin(rotation_mat[2][0]))
    pitch = math.degrees(math.atan2(rotation_mat[2][1], rotation_mat[2][2]))
    roll = math.degrees(math.atan2((rotation_mat[1][0])/math.cos(math.radians(yaw)), rotation_mat[0][0]/math.cos(math.radians(yaw))))

    np_rodrigues = rotation_vector
    rmat = cv2.Rodrigues(np_rodrigues)[0]
    camera_position = -(rmat).T @ (translation_vector)

    tx = camera_position[0][0]
    ty = camera_position[1][0]
    tz = camera_position[2][0]

    return (tx, ty, tz), (pitch, yaw, roll), success

#generates weights for linear regression
def LinRegLossFunc(x_values:list, y_values:list, showGraph:bool):
    """
    x_values:list --> list of x-coordinates
    y_values:list --> list of y-coordinates
    showGraph:bool --> choose to show matplotlib graph
    """

    #MSE Loss Function
    def LossAndGradient(inp, params, truth):

        '''
        inp is input data - numpy array of float types
        params is model parameters - numpy array of float types
        truth is truth data - numpy array of float types
        '''
        #calculating loss
        loss = 0
        for i in range(len(inp)):
            argument = abs(truth[i] - np.dot(params, inp[i]))**2
            loss += argument
        val = 1/len(inp)
        loss *= val

        #calculating gradient
        gradient = 0
        for i in range(len(inp)):
            arg = (np.dot((truth[i] - np.dot(params, inp[i])), inp[i]))
            gradient += arg

        val = -2/(len(inp))
        gradient *= val

        return loss, gradient

    y = np.array(y_values)
    X = np.vstack([np.ones(len(y_values)), x_values]).T
    weights = np.random.normal(size=(2,)) # the two weights parameters

    step_size = 0.0001
    iterations = 4000
    losses = list()

    for _ in range(iterations):
        loss, grad = LossAndGradient(X, weights, y)
        weights -= step_size * grad
        losses.append(loss)

    accuracy = round((1 - min(losses)/max(losses)) * 100)

    if showGraph == True:
        plt.plot(losses)
        plt.show()

    plt.scatter(x_values, y_values)
    x_dat = np.array([min(x_values), max(x_values)])
    y_dat = [weights[0] + weights[1] * x for x in x_dat]
    plt.plot(x_dat, y_dat, c='r')
    if showGraph == True:
        plt.show()
    
    return losses, weights

# if changes made to cpp files, run "cmake . && make && python3 ./pydet.py"
from build.filters import *

class Filters:
    def __init__(self, img_name:str):
        self.img_name = img_name

    
    def saturate(self, satLevel:float, imgout:bool):
        """
        imgout:bool --> choose to save saturated image
        """
        

        img = cv2.imread(self.img_name)
        imout = saturate(img, satLevel)

        if imgout:
            cv2.imwrite("imgSaturate.png", imout)
        
        return imout