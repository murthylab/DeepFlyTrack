
import numpy as np
import cv2
import copy
import sys
if sys.version_info >= (3, 0):
   import tkinter
   from tkinter import filedialog as tkFileDialog
else:
   import Tkinter # python 3: tkinter
   import tkFileDialog # python 3: filedialog
import h5py

import os.path

def updateImage(x, y):
   global arenaX, arenaY, rectRadius, rectCenterX, rectCenterY, showFrame
   
   xdelta = abs(arenaX-x);
   ydelta = abs(arenaY-y);
   delta = max(xdelta,ydelta);

   rectRadius = delta/2;
   rectCenterX = arenaX;
   rectCenterY = arenaY;
   
   img = copy.copy(showFrame);
   cv2.circle(img, (arenaX, arenaY), delta, (0, 255, 0), 3)
   cv2.imshow('display',img)

def areaCallback(event, x, y, flags, param):
   global currentlyDrawing, arenaX, arenaY

   if event == cv2.EVENT_LBUTTONDOWN and currentlyDrawing == False:
      arenaX = x;
      arenaY = y;

   if event == cv2.EVENT_LBUTTONDOWN:
      currentlyDrawing = True;
   elif event == cv2.EVENT_LBUTTONUP:
      currentlyDrawing = False;
   
   if currentlyDrawing:
      updateImage(x,y);

def updateCenterImage(x, y):
   global arenaX, arenaY, showFrame, framePtr, flyPtr, flyData
   
   img = copy.copy(showFrame);
   cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
   if flyPtr < flyData.shape[1]:
      flyData[framePtr,flyPtr] = x
      flyData[framePtr,flyPtr+1] = y
   flyPtr += 2

   cv2.imshow('display',img)

def zoomImage(x, y):
   global arenaX, arenaY, showFrame, framePtr, flyFrames
   
   showFrame = showFrame[x-96:x+96,y-96:y+96]
   flyFrames[framePtr] = showFrame
   # img = copy.copy(showFrame[x-96:x+96,y-96:y+96]);
   # showFrame = img
   # cv2.circle(img, (x, y), 5, (0, 255, 0), 3)
   cv2.imshow('display',showFrame)
   print('displayed')

def centerCallback(event, x, y, flags, param):
   global currentlyDrawing, currentlyZooming, arenaX, arenaY

   if event == cv2.EVENT_LBUTTONDOWN and not currentlyDrawing:
      currentlyDrawing = True;
      updateCenterImage(x,y);

   if event == cv2.EVENT_LBUTTONUP:
      currentlyDrawing = False;

   if event == cv2.EVENT_RBUTTONDOWN and not currentlyZooming:
      currentlyZooming = True;
      # arenaX = x;
      # arenaY = y;
      zoomImage(y,x);
   if event == cv2.EVENT_RBUTTONUP:
      currentlyZooming = False;
      
      

parser = argparse.ArgumentParser()
parser.add_argument('initialdir',type=str,default='.', help='Movie to analyze')
parser.add_argument('-nFlies',dest='nFlies',type=int,default=2, help='Number of flies in arena')
parser.add_argument('-nFeatures',dest='nFeatures',type=int,default=3, help='Number of features per fly')
pargs = parser.parse_args()

# these are the global variables that we want to assign
currentlyDrawing = False;
currentlyZooming = False;
framePtr = 0
flyPtr = 0
arenaX = 0;
arenaY = 0;
delta = 0;
rectCenterX = 0;
rectCenterY = 0;
rectRadius = 0;
angle = 0;
nFlies = 0;
flyCenters = np.zeros((100,2))
fly1CenterX = 0
fly1CenterY = 0
fly2CenterX = 0
fly2CenterY = 0

movieName = tkFileDialog.askopenfilename(initialdir=pargs.initialdir);
outName = movieName.split('/')[-2]
print(outName)
print(movieName)

NumberOfFrames = 150000;
currFrame = 0;

import skvideo.io
useSKV = 1

videoInfo = skvideo.io.ffprobe(movieName)
width = int(videoInfo['video']['@width'])
height = int(videoInfo['video']['@height'])

videodata = skvideo.io.vreader(movieName)

for _ in range(2000):
   frame = next(videodata)

frame = next(videodata)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


print((width,height))
cv2.namedWindow('display')
cv2.resizeWindow('display',width, height)


f = h5py.File(outName + '.hdf5', 'w')
numframes = 300
flyFrames = f.create_dataset('flyFrames', (numframes, 192,192,3), compression='gzip', dtype=np.uint8)
flyData = f.create_dataset('flyPoints', (numframes, pargs.nFlies*2*pargs.nFeatures), compression='gzip')

# choose arena center
print((width,height))
done = False

try:
   while not done:
      flyPtr = 0
      frame2 = copy.copy(frame);
      cv2.putText(frame2, 'Choose area to annotate', (50,50), cv2.FONT_HERSHEY_PLAIN, 2, 255)
      # showFrame = cv2.resize(frame2,(width/2,height/2));
      showFrame = frame2
      cv2.imshow('display',showFrame)
      cv2.setMouseCallback('display',centerCallback)
      keyvalue = cv2.waitKey(0)
      # centerX, centerY = arenaX, arenaY;

      if keyvalue == ord('q'):
         exit()
      else:

         if flyPtr == pargs.nFlies*2*nFeatures:
            framePtr += 1

         else:
            currFlyNum = floor(flyPtr / (nFeatures*2))
            flyData[framePtr,(currFlyNum*(nFeatures*2)):] = 0
            framePtr += 1

         for _ in range(120):
            frame = next(videodata)
         frame = next(videodata)
         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
except:
   pass