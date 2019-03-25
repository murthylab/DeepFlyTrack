import h5py
import cv2
import numpy as np
import os
import scipy.io as sio
import copy
import sys

if sys.version_info >= (3, 0):
   import tkinter
   # import filedialog
   from tkinter import filedialog as tkFileDialog
else:
   import Tkinter # python 3: tkinter
   import tkFileDialog # python 3: filedialog

def getCentroids(movieName):
   analyzedName = movieName[0:-4] + '_centroids.mat'
   # automatedName = movieName[0:-4] + '_cenfixedAngles.mat'
   automatedName = movieName[0:-4] + '_fixedAngles.mat'

   analyzedName0 = movieName[0:-4] + '_tracks.mat'
   automatedName0 = movieName[0:-4] + '_fixedAngles.mat'
   

   # DNNName = movieName[0:-4] + '_anglesDNN3.mat'
   annotationFile = movieName[0:-4] + '_annotated.dat'
   # print('Looking for analyzed data in ' + str(analyzedName) + '...')
   print('Looking for analyzed data in ' + str(automatedName) + '...')
   if (os.path.isfile(analyzedName)):
      print('Found!')
      trackFile = sio.loadmat(analyzedName)
      automatedFile = sio.loadmat(automatedName)
      # DNNFile = sio.loadmat(DNNName)
      if not(os.path.isfile(analyzedName)):
         print('But could not find the annotation file :( You lose.')
         return
   elif os.path.isfile(automatedName0):
      print('Found!')
      trackFile = sio.loadmat(analyzedName0)
      automatedFile = sio.loadmat(automatedName0)
      # DNNFile = sio.loadmat(DNNName)
      if not(os.path.isfile(analyzedName0)):
         print('But could not find the annotation file :( You lose.')
         return
   else:
      print('Could not find track file. Perhaps you chose the wrong movie?')
      return

   return automatedFile['centroids']

def getMovieData(movieName):
   # print(movieName[0:-4] + '_annotated2.mat')
   if os.path.isfile(movieName[0:-4] + '_annotated2.mat'):
      data = sio.loadmat(movieName[0:-4] + '_annotated2.mat')
      width = data['width'][0][0]
      height = data['height'][0][0]
      centerX = data['arenaCenter'][0][0]
      centerY = data['arenaCenter'][0][1]
      radius = data['arenaRadius'][0][0]
      rectCenterX = data['LEDcenter'][0][0]
      rectCenterY = data['LEDcenter'][0][1]
      rectRadius = data['LEDradius'][0][0]
   else:
      f=open(movieName[0:len(movieName)-4] + '_annotated.dat','r');
      width = int(f.readline())
      height = int(f.readline())
      centerX = int(f.readline())
      centerY = int(f.readline())
      radius = float(f.readline())
      rectCenterX = int(f.readline())
      rectCenterY = int(f.readline())
      rectRadius = float(f.readline())
      angle = float(f.readline())
      f.close()

   arenaCoords = (max(0,centerX-radius), max(0,centerY-radius), min(centerX+radius,width), min(centerY+radius,height));

   return arenaCoords

def newFrameCallback(position):
   global cap, frameInd, centroids
   frameInd = position
   cap.set(cv2.CAP_PROP_POS_FRAMES, frameInd)
   _,frame = cap.read()


   frame = addHUD(frame)
   cv2.imshow('display',frame/255)

def clickCallback(event, x, y, flags, param):
   global cap, frameInd, assignEvents

   if event == cv2.EVENT_LBUTTONUP:
      cap.set(cv2.CAP_PROP_POS_FRAMES, frameInd)
      _,frame = cap.read()


      # figure out closest centroid
      closestFly = np.argmin(((centroids[frameInd,:,1]+arenaCoords[0])/zoom - x)**2 + ((centroids[frameInd,:,0]+arenaCoords[1])/zoom - y)**2)

      # swap that with newCentroid from here until the next time point that this fly has been selected
      assignEvents[frameInd,currFly] = 1
      nextEvents = np.where(assignEvents[:,currFly] != 0)[0]
      nextEvents = nextEvents[nextEvents - frameInd > 0]
      if nextEvents.size == 0:
         newCentroids[frameInd:,currFly,:] = centroids[frameInd:,closestFly,:]
      else:
         newCentroids[frameInd:nextEvents[0],currFly,:] = centroids[frameInd:nextEvents[0],closestFly,:]

      frame = addHUD(frame)
      cv2.imshow('display',frame/255)

def addHUD(frame):
   global frameInd, centroids, newCentroids

   frame = copy.copy(frame)
   # print(frame.shape)
   width = frame.shape[0]
   height = frame.shape[1]
   frame = cv2.resize(frame,(int(height/zoom),int(width/zoom)))
   cv2.putText(frame, 'working on fly ' + str(currFly+1), (50,50), cv2.FONT_HERSHEY_PLAIN, 2, 255)
   for ii in list(range(centroids.shape[1])):
      color = [155,155,155]
      cv2.circle(frame,(int((centroids[frameInd,ii,1]+arenaCoords[0])/zoom),int((centroids[frameInd,ii,0]+arenaCoords[1])/zoom)),4,color,-1)
      for hh in list(range(max(frameInd-30,0),frameInd)):
         cv2.circle(frame,(int((centroids[hh,ii,1]+arenaCoords[0])/zoom),int((centroids[hh,ii,0]+arenaCoords[1])/zoom)),1,color,-1)


   color = [0,0,255]
   cv2.circle(frame,(int((newCentroids[frameInd,currFly,1]+arenaCoords[0])/zoom),int((newCentroids[frameInd,currFly,0]+arenaCoords[1])/zoom)),4,color,-1)
   for hh in list(range(max(frameInd-30,0),frameInd)):
      cv2.circle(frame,(int((newCentroids[hh,currFly,1]+arenaCoords[0])/zoom),int((newCentroids[hh,currFly,0]+arenaCoords[1])/zoom)),1,color,-1)

   return frame

# design: read in an HDF5 file
# use a little tic bar on the bottom to scroll through frame
# start with fly 1. Annotate through the whole movie which fly is 1
# Then go through next fly (should only have to do two)

parser = argparse.ArgumentParser()
parser.add_argument('initialdir',type=str,default='.', help='Movie to analyze')
pargs = parser.parse_args()

dirName = tkFileDialog.askdirectory(initialdir=pargs.initialdir)

if dirName[-1] != '/':
   dirName += '/'

mname = dirName.split('/')[-2]

centroids = getCentroids(dirName + mname + '.avi')
arenaCoords = getMovieData(dirName + mname + '.avi')

newCentroids = copy.copy(centroids)
assignEvents = np.zeros((centroids.shape[0],centroids.shape[1]))

zoom = 2

# https://github.com/scikit-video/scikit-video/issues/30
# with h5py.File(dirName + 'videofile.hdf5', 'r') as f:
frameInd = 0
#    frame = f['frames'][frameInd]
# TS3 -> seekable
cap = cv2.VideoCapture(dirName + mname + '_seekable.mp4')
print(dirName + mname + '_seekable.mp4')
# totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
totalFrames = centroids.shape[0]
cap.set(cv2.CAP_PROP_POS_FRAMES, frameInd)
_,frame = cap.read()

currFly = 0

# align microphone 1
frame = addHUD(frame)

cv2.imshow('display',frame/255)
cv2.createTrackbar('frameNum','display',frameInd,totalFrames,newFrameCallback)
cv2.setMouseCallback('display',clickCallback)

while 1:
   key = cv2.waitKey(0)
   if key == ord('q'):
      print('quitting...')
      break
   if key == ord('n'):
      frameInd += 10
      if frameInd >= totalFrames:
         frameInd = totalFrames - 1
   if key == ord('p'):
      frameInd -= 10
      if frameInd < 0:
         frameInd = 0
   elif key == ord('1'):
      currFly = 0
   elif key == ord('2'):
      currFly = 1
   elif key == ord('3'):
      currFly = 2

   cap.set(cv2.CAP_PROP_POS_FRAMES, frameInd)
   _,frame = cap.read()
   frame = addHUD(frame)
   cv2.setTrackbarPos('frameNum','display',frameInd)
   cv2.imshow('display',frame/255)

savefile = dirName + 'handfixedCentroids.mat'
sio.savemat(savefile,{'fixedCentroids':newCentroids,'fixedEvents':assignEvents})
