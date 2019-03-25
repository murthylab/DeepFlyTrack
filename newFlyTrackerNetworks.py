from keras.models import load_model
import skvideo.io
import numpy as np
import cv2
import os

import scipy.io as sio
from skimage.transform import rescale

from newFlyTrackerUtils import *

def getPredictionModels(arenaName):
	# centroidModel = load_model('models/' + arenaName + '_centroids.h5')
	# centroidModel = load_model('models/flyPixelsnew_adam_1.0_augmentor__hourglass_2.0_2_False__autoSave.h5')
	# centroidModel = load_model('models/flyPixelsnew_adam_1.0_augmentor__hourglass_2.0_5_False__autoSave.h5')
	# centroidModel = load_model('models/flyPixelsnew_adam_1.0_augmentor__hourglass_1.0_5_False__autoSave.h5')



	# centroidModel = load_model('models/flyPixelsnew_adam_1.0_augmentor__hourglass_4.0_3_False__autoSave.h5')
	# centroidModel = load_model('models/flyPixelsnew_adam_1.0_augmentor__hourglass_2.0_5_False__autoSave (2).h5')
	# centroidModel = load_model('models/flyPixelsnew_adam_1.0_augmentor__hourglass_4.0_3_False__autoSave.h5')
	# centroidModel = load_model('models/flyPixelsnew_adam_0.25_augmentor__hourglass_2.0_3_False__autoSave.h5')
	# centroidModel = load_model('models/flyPixelsnew_adam_1.0_augmentor__gray__hourglass_1.0_5_False__autoSave.h5')
	if arenaName == '':
		centroidModel = load_model('models/flyPixelsnew_adam_1.0_augmentor__gray__hourglass_1.0_5_False__autoSave.h5')
	else:
		centroidModel = load_model('models/flyPixelsnew_' + arenaName + '_adam_1.0_augmentor__gray__hourglass_1.0_5_False__autoSave.h5')

	return centroidModel

def predictBackground(movieName,model):
	videoInfo = skvideo.io.ffprobe(movieName)
	width = int(videoInfo['video']['@width'])
	height = int(videoInfo['video']['@height'])
	totalFrames = np.int(videoInfo['video']['@nb_frames'])
	fps = videoInfo['video']['@avg_frame_rate'].split('/')
	if len(fps) > 1:
		fps = int(fps[0])/int(fps[1])
	else:
		fps = int(fps)

	frameCoords, _ = getMovieData(movieName)
	totalBkg = 10
	numBgFrames = 10

	keyFrames = np.linspace(0,totalFrames,totalBkg-2).astype(int)
	print(keyFrames)

	rdr2 = skvideo.io.vreader(movieName)
	frame = next(rdr2)

	print(frameCoords)
	# frameCoords[0] += 708 - 96
	# frameCoords[1] += 501
	# frameCoords[3] = frameCoords[1] + 192
	# frameCoords[2] = frameCoords[0] + 192
	# frameCoords = (frameCoords[0] + 708 - 96, frameCoords[1] + 501 - 96, frameCoords[0] + 708 + 96, frameCoords[1] + 501 + 96)
	arenaFrame = frame[frameCoords[1]:frameCoords[3],frameCoords[0]:frameCoords[2]] *255
	# arenaFrame = rescale(arenaFrame,1.0 / 1.0)
	print(np.max(arenaFrame))
	print(arenaFrame.shape)

	bgAccumulateAll = np.zeros((totalBkg,arenaFrame.shape[0], arenaFrame.shape[1]))
	bgAccumulateMinute = np.zeros((numBgFrames,arenaFrame.shape[0], arenaFrame.shape[1]))
   
	keyNum = 0
	framesToUse = np.linspace(keyFrames[0],keyFrames[1]-1,numBgFrames).astype(int)
	print(framesToUse)
	thisInd = 0
	try:
		for frameInd,frame in enumerate(rdr2):
			if frameInd >= keyFrames[keyNum+1]:
				print('new key!')
				bgAccumulateAll[keyNum] = np.median(bgAccumulateMinute,axis=0)
				# sio.savemat('bkg_' + str(keyNum) + '.mat',{'bkg':np.median(bgAccumulateMinute,axis=0)})
				keyNum += 1
				if keyNum == len(keyFrames):
					break

				framesToUse = np.linspace(keyFrames[keyNum],keyFrames[keyNum+1]-1,numBgFrames).astype(int)
				print(framesToUse)
				thisInd = 0

			if frameInd in framesToUse:
				# print(frameInd)
				frame = frame[:,:,2::-1]
				frame = 1-np.expand_dims(frame[:,:,1],axis=-1)
				# print(np.mean(frame[50:750,50:750,1]))
				# print(np.mean(frame[50:750,50:750,0]))
				# print(np.mean(frame[50:750,50:750,2]))
				# arenaFrame = model.predict(np.expand_dims(rescale(frame[frameCoords[1]:frameCoords[3],frameCoords[0]:frameCoords[2]]*255,1.0/1.0),axis=0))[0,:,:,0]
				arenaFrame = model.predict(np.expand_dims(frame[frameCoords[1]:frameCoords[3],frameCoords[0]:frameCoords[2]]*255,axis=0))[0,:,:,0]
	 			bgAccumulateMinute[thisInd] = arenaFrame
	 			# sio.savemat('bkg_' + str(keyNum) + '_' + str(thisInd) + '.mat',{'bkg':bgAccumulateMinute[thisInd]})
	 			# sio.savemat('bkgtmp_' + str(keyNum) + '_' + str(thisInd) + '.mat',{'frg':np.expand_dims(rescale(frame[frameCoords[1]:frameCoords[3],frameCoords[0]:frameCoords[2]]*255,1.0/1.0),axis=0)})
	 			thisInd += 1
	except:
		pass

	background = np.median(bgAccumulateAll,axis=0)
	return background

def getFlyContours(img, nFlies=2):
   minFlyArea = 10
   # minFlyAreaNorm = 0.001
   minFlyAreaNorm = 0.0005
   # maxFlyAreaNorm = 0.02

   # cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   _,contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   area = [0]*len(contours);
   for idx,cnt in enumerate(contours):
      area[idx] = cv2.contourArea(cnt);

   arenaArea = abs(img.shape[1]*img.shape[0])
   index = 0
   print(area)
   areaNorm = np.sort(np.array([ai/arenaArea for ai in area]))
   print(areaNorm)
   # if areaNorm.shape[0] > 1:
   # 		minFlyAreaNorm = areaNorm[-2]
   if areaNorm.shape[0] > 1:
      minFlyAreaNorm = np.maximum(areaNorm[-np.min([nFlies,areaNorm.shape[0]])],minFlyAreaNorm)

   print(minFlyAreaNorm)
   for cntInd,cnt in enumerate(contours):
      if area[cntInd]/arenaArea >= minFlyAreaNorm:
         contours[index]=contours[cntInd]
         index = index + 1;
   contours = contours[0:index]

   return contours

def ellipseListToArray(ellipse):
   return np.float32([ellipse[0][0],ellipse[0][1],ellipse[1][0],ellipse[1][1],ellipse[2]])

def predictCentroids(frame, radius, model, nFlies=2, bkg=None, dumpDir=None):
	# pass frame through - network depends on the arena
	# do some bluring and return potential bounding boxes
	# use opencv to clear circle of frame, either before or after passing through network?
	# then blur
	# should we subtract a background?
	if dumpDir == '':
		dumpDir = None

	if bkg is None:
		bkg = np.zeros(frame.shape)

	# print('aha')
	# frame = frame[:,:,2::-1]
	# print(frame.shape)
	frame = (1-np.expand_dims(frame[:,:,:,1],axis=-1))*255
	# print('aww')
	# print(frame.shape)

	# print(frame.shape)
	prediction = model.predict(frame)
	# print('?')
	if dumpDir is not None:
		cv2.imwrite(dumpDir + 'frame.png',frame[0,:,:,0])
		cv2.imwrite(dumpDir + 'pred.png',np.uint8((prediction[0,:,:,0])*255*1))
		cv2.imwrite(dumpDir + 'raw.png',np.uint8((prediction[0,:,:,0] - bkg)*255*1))
		cv2.imwrite(dumpDir + 'pos.png',(prediction[0,:,:,0] - bkg > 0.05)*255)
		cv2.imwrite(dumpDir + 'neg.png',(prediction[0,:,:,0] - bkg < 0)*255)
	pred_orig = prediction[0,:,:,0]
	prediction = prediction[0,:,:,0] - bkg
	prediction[prediction < 0] = 0
	# sio.savemat('pred.mat',{'pred':prediction,'pred0':pred_orig,'bkg':bkg,'frame':frame})

	mask = np.zeros(prediction.shape,np.uint8)
	# print(prediction.shape)
	# print(radius)
	cv2.circle(mask,(int(prediction.shape[0]/2),int(prediction.shape[1]/2)),radius,255,-1)

	prediction[mask!=255] = 0;

	prediction = cv2.GaussianBlur(np.uint8(prediction*255*6),(11,11),5)
	# print(np.max(prediction))
	prediction = (prediction > 60)*255
	# cv2.imwrite('pred_.jpg',np.uint8(prediction))
	prediction = cv2.blur(prediction,(3,3))
	prediction = cv2.blur(prediction,(3,3))
	prediction = cv2.blur(prediction,(3,3))
	if dumpDir is not None:
		cv2.imwrite(dumpDir + 'pred_.jpg',np.uint8(prediction*255*10))
	prediction = (prediction > 150)*255

	# now do some denoising (k-means)
	contours = getFlyContours(np.uint8(prediction),nFlies)
	markers = np.zeros((prediction.shape[0],prediction.shape[1],1),np.int32)
	for cntInd,cnt in enumerate(contours):
		cv2.drawContours(markers,contours,cntInd,cntInd+1,-1)

	fgObjects = np.zeros(prediction.shape,np.uint8)
	fgObjects[np.logical_and(markers[:,:,0] > 0, prediction > 0)] = prediction[np.logical_and(markers[:,:,0] > 0, prediction > 0)]
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	cv2.erode(fgObjects,kernel2,fgObjects)
	pts = np.vstack(np.nonzero(fgObjects)).astype(np.float32).T
	pts = pts[:,0:2]
	numK = nFlies
	term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
	bestLabelsKM = np.zeros(len(pts))
	compactness, bestLabelsKM, centers = cv2.kmeans(pts, numK, bestLabelsKM, term_crit, 10, cv2.KMEANS_PP_CENTERS)


	bodyEllipses = np.zeros((2, 5), np.float16);
	allRedLines = np.zeros((2, 4), np.float16);
	ind = bestLabelsKM==0;
	ptsBody = np.vstack((pts[ind[:,0],0], pts[ind[:,0],1])).T;
	flyEllipse = cv2.fitEllipse(ptsBody[:,::-1])
	bodyEllipses[0] = ellipseListToArray(flyEllipse)

	thisLine = cv2.fitLine(ptsBody, cv2.DIST_L2, 0, 0.01, 0.01)
	allRedLines[0] = np.squeeze(thisLine)

	ind = bestLabelsKM==1;
	ptsBody = np.vstack((pts[ind[:,0],0], pts[ind[:,0],1])).T;
	flyEllipse = cv2.fitEllipse(ptsBody[:,::-1])
	bodyEllipses[1] = ellipseListToArray(flyEllipse)

	thisLine = cv2.fitLine(ptsBody, cv2.DIST_L2, 0, 0.01, 0.01)
	allRedLines[1] = np.squeeze(thisLine)

	if dumpDir is not None:
		tmp = np.zeros((prediction.shape[0],prediction.shape[1],3))
		ind = bestLabelsKM==0;
		pts2 = np.vstack((pts[ind[:,0],0], pts[ind[:,0],1])).T;
		for (x,y) in pts2:
			tmp[int(x),int(y),0] = 255;
		ind = bestLabelsKM==1;
		pts2 = np.vstack((pts[ind[:,0],0], pts[ind[:,0],1])).T;
		for (x,y) in pts2:
			tmp[int(x),int(y),1] = 255;
		ind = bestLabelsKM>=2;
		pts2 = np.vstack((pts[ind[:,0],0], pts[ind[:,0],1])).T;
		for (x,y) in pts2:
			tmp[int(x),int(y),2] = 255;

		cv2.imwrite(dumpDir + 'kMeans_' + str(numK) + '.jpg',tmp)

	return prediction, centers, bodyEllipses, allRedLines

def predictOrientationMap(frame, model):
	# pass frame through - network depends on the arena
	pass

def predictArenaInfo(frame,movieName):
	# predict information about where the arena is located, where the blinking light is, etc
	# return blinkCoords,frameCoords

	# cheat right now and just load the info file...
	return getMovieData(movieName)

def getMovieData(movieName):
	if os.path.isfile(movieName[0:len(movieName)-4] + '_annotated2.mat'):
		data = sio.loadmat(movieName[0:len(movieName)-4] + '_annotated2.mat')
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

	arenaCoords = (int(max(0,centerX-radius)), int(max(0,centerY-radius)), int(min(centerX+radius,width)), int(min(centerY+radius,height)))
	blinkyCoords = (int(max(0,rectCenterX-rectRadius)), int(max(0,rectCenterY-rectRadius)), int(min(rectCenterX+rectRadius,width)), int(min(rectCenterY+rectRadius,height)))
   
	return arenaCoords, blinkyCoords