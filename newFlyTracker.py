
# new fly tracker
import numpy as np
import argparse
import skvideo.io
import cv2
import scipy.io as sio
import os
import h5py

from newFlyTrackerNetworks import *
from newFlyTrackerUtils import *

import time

parser = argparse.ArgumentParser()
parser.add_argument('filename',type=str,default='', help='Movie to analyze')
parser.add_argument('-arena',dest='arenaName',type=str,default='', help='Choose which arena settings to use')
parser.add_argument('-debugText',dest='debugText',action='store_true', help='Display debug text')
parser.add_argument('-blinky',dest='blinky',action='store_true', help='Only analyze blink state')
parser.add_argument('-grabFrames',dest='grabFrames',action='store_true', help='Grab frames surrounding flies for future use')
parser.add_argument('-frames',dest='frames',nargs=2, action='store', type=int, help='Only analyze frames [n1 n2]. Note that these can only be APPROXIMATELY guaranteed.')
parser.add_argument('-dump',dest='dumpdir',type=str,default='', help='Dump processed images to a directory')
parser.add_argument('-nFlies',dest='nFlies',type=int,default=2, help='Number of flies in arena')
parser.add_argument('-startFrame',dest='startFrame',type=int,default=0, help='Don''t track movie until this frame')
parser.add_argument('-useFrameFile',dest='useFrameFile',action='store_false', help='Use the startTracking file to choose which frame to start tracking on')
pargs = parser.parse_args()

def getStartTime(dirName):
   trackingFile = dirName + 'StartTrackingFrame.txt'
   if (os.path.isfile(trackingFile)):
      f=open(trackingFile)
      startFrame = int(f.readline())
      print('starting frame is ' + str(startFrame))
      f.close()

      return startFrame
   else:
      return 1

if __name__ == '__main__':
	# point to movie
	print(pargs.filename)
	movieName = pargs.filename

	# start everything after a certain number of frames
	if pargs.frames is not None:
		startFrame = pargs.frames[0]
	elif pargs.useFrameFile:
		startFrame = getStartTime(movieName.split('/')[0])
	else:
		startFrame = 0

	print('loading models...')
	centroidModel = getPredictionModels(pargs.arenaName)
	print('predicting background...')
	# can always save the background...
	if not os.path.exists(movieName[:-4] + '_bkg.mat'):
		background = predictBackground(movieName,centroidModel,startFrame)
		sio.savemat(movieName[:-4] + '_bkg.mat',{'background':background})
	else:
		background = sio.loadmat(movieName[:-4] + '_bkg.mat')['background']

	print('loading movie...')
	videoInfo = skvideo.io.ffprobe(movieName)
	width = int(videoInfo['video']['@width'])
	height = int(videoInfo['video']['@height'])
	videodata = skvideo.io.vreader(movieName)
	print(width)
	print(height)
	frame = next(videodata)

	if pargs.grabFrames:
		f = h5py.File(movieName[:-4] + '_frames.hdf5','w')

		flyFrames = f.create_dataset('frames', (10000, 192, 192,3), compression="gzip", dtype=np.uint8)
		frameNums = f.create_dataset('frameNum', (10000, ), compression="gzip")
		fliesInFrames = f.create_dataset('fliesInFrames', (10000, ), compression="gzip")

	flyInd = 0

	centroidList = []
	bodyEllipseList = []
	flyLinesList = []
	blinkList = []

	# try:
	for frameInd,frame in enumerate(videodata):
		if pargs.frames is not None:
			if frameInd < pargs.frames[0]:
				print(frameInd)
				continue
			elif frameInd > pargs.frames[1]:
				print('bye bye')
				break

		t = time.time()
		frame = rgb2bgr(frame)
		if frameInd == startFrame:
			frameCoords,blinkCoords = predictArenaInfo(frame,movieName)

		# if frameInd < 519:
		# 	continue

		# extract blinking light + behavior part of each frame
		# print(np.sum(frame))
		# print(blinkCoords)
		# print(frameCoords)
		print(frameInd)
		blinkFrame = frame[blinkCoords[1]:blinkCoords[3],blinkCoords[0]:blinkCoords[2]]
		behaveFrame = frame[frameCoords[1]:frameCoords[3],frameCoords[0]:frameCoords[2]]
		# behaveFrame = rescale(behaveFrame,1.0 / 1.0)
		radius = (frameCoords[3]-frameCoords[1])/2

		# find blinking light intensity
		# blinkState[frameInd] = detectBlinkState(blinkFrame)

		# find potential fly centroids/bounding boxes
		centroidBox,centroids,bodyEllipses,flyLines = predictCentroids(np.expand_dims(behaveFrame,axis=0), radius, centroidModel, nFlies=pargs.nFlies, bkg=background,dumpDir=pargs.dumpdir)
		print('that prediction took ' + str(time.time() - t) + ' (' + str(centroids) + ')')
		centroidList.append(centroids)
		bodyEllipseList.append(bodyEllipses)
		flyLinesList.append(flyLines)

		blinkList.append(np.sum(blinkFrame))


		if pargs.grabFrames:
			centroidDist = np.sqrt(np.sum((centroids[0] - centroids[1])**2))
			# print([int((centroids[0][0] + centroids[1][0])/2),int((centroids[0][1] + centroids[1][1])/2)])
			# print([int(centroids[0][0]),int(centroids[0][1])])
			# print([int(centroids[1][0]),int(centroids[1][1])])
			# print(np.max(behaveFrame),np.min(behaveFrame))
			print(centroids)
			print(behaveFrame.shape)
			# print(behaveFrame.shape)
			if (centroidDist > 192/3):
				# take two boxes with centered flies
				# print(behaveFrame[int(centroids[0][0])-96:int(centroids[0][0])+95,int(centroids[0][1])-96:int(centroids[0][1]+96)].shape)
				# print(np.max(behaveFrame[int(centroids[0][0])-96:int(centroids[0][0])+96,int(centroids[0][1])-96:int(centroids[0][1]+96),:]))
				if int(centroids[0][0])-96 < 0:
					fxOffset = int(centroids[0][0])-96
				elif int(centroids[0][0])+96 >= behaveFrame.shape[0]:
					fxOffset = behaveFrame.shape[0] - int(centroids[0][0])+96
				else:
					fxOffset = 0
				if int(centroids[0][1])-96 < 0:
					fyOffset = int(centroids[0][1])-96
				elif int(centroids[0][1])+96 > behaveFrame.shape[1]:
					fyOffset = int(centroids[0][1])+96 - behaveFrame.shape[1]
				else:
					fyOffset = 0

				flyFrames[flyInd] = behaveFrame[int(centroids[0][0])-96-fxOffset:int(centroids[0][0])+96-fxOffset,int(centroids[0][1])-96-fyOffset:int(centroids[0][1])+96-fyOffset,:]
				frameNums[flyInd] = frameInd
				fliesInFrames[flyInd] = 1
				flyInd = flyInd + 1

				if int(centroids[1][0])-96 < 0:
					fxOffset = int(centroids[1][0])-96
				elif int(centroids[1][0])+96 >= behaveFrame.shape[0]:
					fxOffset = behaveFrame.shape[0] - int(centroids[1][0])+96
				else:
					fxOffset = 0
				if int(centroids[1][1])-96 < 0:
					fyOffset = int(centroids[1][1])-96
				elif int(centroids[1][1])+96 > behaveFrame.shape[1]:
					fyOffset = int(centroids[1][1])+96 - behaveFrame.shape[1]
				else:
					fyOffset = 0

				flyFrames[flyInd] = behaveFrame[int(centroids[1][0])-96-fxOffset:int(centroids[1][0])+96-fxOffset,int(centroids[1][1])-96-fyOffset:int(centroids[1][1]+96-fyOffset),:]
				frameNums[flyInd] = frameInd
				flyInd = flyInd + 1
				fliesInFrames[flyInd] = 1

			else:
				# take one box with both flies in the box
				if int((centroids[0][0] + centroids[1][0])/2)-96 < 0:
					fxOffset = int((centroids[0][0] + centroids[1][0])/2)-96
				elif int((centroids[0][0] + centroids[1][0])/2)+96 >= behaveFrame.shape[0]:
					fxOffset = behaveFrame.shape[0] - int((centroids[0][0] + centroids[1][0])/2)+96
				else:
					fxOffset = 0
				if int((centroids[0][1] + centroids[1][1])/2)-96 < 0:
					fyOffset = int((centroids[0][1] + centroids[1][1])/2)-96
				elif int((centroids[0][1] + centroids[1][1])/2)+96 > behaveFrame.shape[1]:
					fyOffset = int((centroids[0][1] + centroids[1][1])/2)+96 - behaveFrame.shape[1]
				else:
					fyOffset = 0

				flyFrames[flyInd] = behaveFrame[int((centroids[0][0] + centroids[1][0])/2)-96-fxOffset:int((centroids[0][0] + centroids[1][0])/2)+96-fxOffset,int((centroids[0][1] + centroids[1][1])/2)-96-fyOffset:int((centroids[0][1] + centroids[1][1])/2)+96-fyOffset]
				frameNums[flyInd] = frameInd
				fliesInFrames[flyInd] = 2
				flyInd = flyInd + 1

			# print(flyInd,flyFrames.shape[0])
			if flyInd > flyFrames.shape[0]:
				# if flyInd > 4:
				break

		# print(centroidBox.shape)
		print('that loop took ' + str(time.time() - t))

		if np.mod(frameInd,100) == 0:
			sio.savemat(movieName[:-4] + '_centroids.mat',{'centroids':centroidList,'blinker':blinkList,'flyLines':flyLinesList,'flyEllipses':bodyEllipseList})
			if pargs.grabFrames:
				f.flush()

		# centroidBox[0] = centroidBox[0] > 0.05
		
		# centroidBox = np.uint8(centroidBox*255*6)
		# print(np.max(centroidBox))
		# print(np.min(centroidBox))
		# cv2.imwrite('dump/pred' + str(frameInd) + '.png',np.uint8(centroidBox))
		# cv2.imwrite('dump/pred' + str(frameInd) + '.png',np.uint8(centroidBox*255*6))
		# cv2.imwrite('dump/pred' + str(frameInd) + '.png',centroidBox[0])
		# cv2.imwrite('dump/data' + str(frameInd) + '.png',behaveFrame)

		# if frameInd > 30:
		# 	exit()


		# # save every so often
		# saveData()

		# okay now we have potential information across all frames... analyze them!
		
		# guess fly identity
		# flyAngles, flyCentroids = fixFlyIdentity(angle, centroid)
		
		# other post-hoc stuff?

		# saveData()
	# except:
	# 	sio.savemat(movieName[:-4] + '_centroids.mat',{'centroids':centroidList,'blinker':blinkList,'flyLines':flyLinesList,'flyEllipses':bodyEllipseList})
	# 	pass


	if pargs.grabFrames:
		f.flush()
		f.close()