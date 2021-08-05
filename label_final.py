######Final Codes

import cv2
import numpy as np
import math
from os import listdir
from os.path import isfile, join


hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0
temp = 0
circleout = 0
########################################
######### only change this part ########

pathforsave = './labels' #path for save the labeled picture 
pathforimg = './resized' #path for read the picture 

mid = 1.2 # for gamma level

######### only change this part ########
########################################


def nothing(x):
    pass
    
def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        
def incgamma(image):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    mean = np.mean(val)
    gamma = 6
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    inputt = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
    
    return inputt

def createwindow(screen):
    
    cv2.namedWindow(screen,cv2.WINDOW_FREERATIO)
    #qqqqqcv2.resizeWindow(screen, 400,800)
    cv2.setMouseCallback(screen, mouse_crop)
    cv2.createTrackbar('HMin',screen,0,179,nothing) # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin',screen,0,255,nothing)
    cv2.createTrackbar('VMin',screen,0,255,nothing)
    cv2.createTrackbar('HMax',screen,0,179,nothing)
    cv2.createTrackbar('SMax',screen,0,255,nothing)
    cv2.createTrackbar('VMax',screen,0,255,nothing)
    cv2.createTrackbar('GMax',screen,1,20,nothing)
    cv2.createTrackbar(savepicname[n],screen,1,2,nothing)
    cv2.setTrackbarPos('HMax',screen, 179)
    cv2.setTrackbarPos('SMax',screen, 255)
    cv2.setTrackbarPos('VMax',screen, 255)
    cv2.setTrackbarPos('GMax',screen, 6)
    cv2.setTrackbarPos(savepicname[n],screen, 1)

def newPhotoWithTrackbars(image,screen):
    hMin = cv2.getTrackbarPos('HMin',screen)
    sMin = cv2.getTrackbarPos('SMin',screen)
    vMin = cv2.getTrackbarPos('VMin',screen)
    hMax = cv2.getTrackbarPos('HMax',screen)
    sMax = cv2.getTrackbarPos('SMax',screen)
    vMax = cv2.getTrackbarPos('VMax',screen)
    gMax = cv2.getTrackbarPos('GMax',screen)
    name = cv2.getTrackbarPos(savepicname[n],screen)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(image)
    mean = np.mean(val)
    if gMax==0:
        gMax=1
    mid = gMax/1.0
    gamma = 1.0
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    inputt = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(inputt,inputt, mask= mask)



    return output

savepicname = [ f for f in listdir(pathforimg) if isfile(join(pathforimg,f)) ]
images = np.empty(len(savepicname), dtype=object)

for n in range(0, len(savepicname)):
    x_start = 0
    y_start = 0
    x_end = 0
    y_end = 0
    x_cs=x_start
    y_cs=y_start
    x_ce=x_end
    y_ce=y_end
    cropping = False
    images[n] = cv2.imread( join(pathforimg,savepicname[n]) )
    inputt = images[n]
    createwindow('image')
    wait_time = 3
    originalimage = images[n]
    val=True
    createwindow('img_orj')
    cv2.imshow('img_orj',originalimage)
    out_bin=1
    pval=False
    print(savepicname[n])
    while(1):
        key= cv2.waitKey(wait_time)

        output=newPhotoWithTrackbars(inputt,'image')      
        cv2.circle(output,(round((x_start+x_end)/2),round((y_start+ y_end)/2)) ,round((((x_start-x_end)**2+(y_start- y_end)**2)**(1/2))/2) ,  (0, 0, 0), -1)
      
        cv2.imshow('image',output)
        pval=False        

        if key == ord('p'):
            pval=True
            inputt=(255- output)
                        
        elif key == ord('r'):
            inputt=originalimage
            
        elif key == ord('k'):
            inputt=output
            
            
        elif key == ord('s'):
            dim = (400,400)
            output=output*out_bin
            val==True
            outcircled = circleout+output
            rgb = cv2.cvtColor(outcircled, cv2.COLOR_HSV2RGB)	
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            gray[gray>0]=255
            resized=cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(join(pathforsave,savepicname[n]), resized)
            
            break
        
        elif key == ord('q'):
            break
        elif key == ord('e'):
            output[output>150]=255
            inputt=output
        
        elif key == ord('t'):
            createwindow('imageforcrop')
 
            temp= newPhotoWithTrackbars(originalimage,'imageforcrop')-output
            temp_orj=temp
            out_bin=output
            out_bin[out_bin>0]=1
            temp_bin=1-out_bin
            x_cs=x_start
            y_cs=y_start
            x_ce=x_end
            y_ce=y_end
            val=False
            
            while(1):
                key2= cv2.waitKey(3)
                output2=newPhotoWithTrackbars(temp,'imageforcrop')
                cv2.circle(output2,(round((x_start+x_end)/2),round((y_start+ y_end)/2)) ,round((((x_start-x_end)**2+(y_start- y_end)**2)**(1/2))/2) ,  (0, 0, 0), -1)
                cv2.imshow('imageforcrop',output2)
                if key2 == ord('p'):
                    temp=(255-output2)
                elif key2 == ord('e'):
                    output2[output2>150]=255
                    temp=output2
                elif key2 == ord('r'):
                    temp=temp_orj
                elif key2 == ord('k'):
                    temp=output2
                if key2 == ord('s'):
                    dim = (400,400)
                    circleout= output2*temp_bin
                    cv2.destroyWindow('imageforcrop') 
                    break
    cv2.destroyAllWindows()
