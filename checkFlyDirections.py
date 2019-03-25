import numpy as np
import os
import fnmatch
import h5py

import numpy as np
# HACK MONKEYPATCH
try:
    import tensorflow as tf
    tf.python.control_flow_ops = tf
except:
    pass
# HACK MONKEYPATCH
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

import h5py
# import skvideo.io
from scipy.io import loadmat
from keras.losses import mean_squared_error
import cv2

import numpy as np
import scipy.io as sio
import sys
import glob
import pandas as pd

from skimage.color import rgb2gray


dirName = sys.argv[1]



filename = glob.glob(dirName + '/*_tracks.mat')[0]
filename = filename.replace('\\','/')

globals().update(sio.loadmat(filename))
print('loaded file')

# # flipFile = '/jukebox/murthy/adamjc/PreProcessed/Flips.xlsx'
# flipFile = 'C:/Users/adamjc/Dropbox/MurthyLab/OpenProjects/GenerativeModel/Flips.xlsx'
# xls = pd.ExcelFile(flipFile)
# sheetX = xls.parse(0)

# fileID = filename[(filename.rfind('/')+1):]
# if fileID[8] == '-':
#     fileID = fileID[:15]
# else:
#     fileID = fileID[:13]

# print(sheetX.index[sheetX['Filename'] == fileID].tolist())
# foundFile = sheetX.index[sheetX['Filename'] == fileID].tolist()
# if foundFile is None:
#     print('could not find file ID ' + fileID + ' in flips spreadsheet!')
#     exit()

# flipFrames = sheetX['FlipFrame'][foundFile].tolist()
# maleIndex = sheetX['Male'][foundFile].tolist()[0]

# print('parsed flips XLSX, found flip frames ' + str(flipFrames))

# # make sure this works for NaN values...
# if not np.any(np.isnan(flipFrames)):
#     if len(flipFrames) > 1:
#         flipFrames = flipFrames[0].split(';')
#         flipFrames = [int(frameNum) for frameNum in flipFrames]

#     newLines = np.zeros(flyLines.shape)
#     # flynum 0 is female, flynum 1 is male
#     maleNum = maleIndex-1
#     femaleNum = int(not maleIndex)
#     flipFrames = np.append(flipFrames,flyLines.shape[0])
#     flipFrames = np.insert(flipFrames,0,0)
#     for ii,flip in enumerate(flipFrames[:-1]):
#         newLines[flipFrames[ii]:flipFrames[ii+1],0,:] = flyLines[flipFrames[ii]:flipFrames[ii+1],femaleNum,:]
#         newLines[flipFrames[ii]:flipFrames[ii+1],1,:] = flyLines[flipFrames[ii]:flipFrames[ii+1],maleNum,:]

#         maleNum = int(not maleNum)
#         femaleNum = int(not femaleNum)

#     flyLines = newLines


fixedAngles = np.zeros((2,len(angles)))
for flynum in range(2):
    degreeVec = np.arange(-360,540,180);

    angle1 = np.zeros(len(flyLines))
    # angle2 = np.zeros(len(flyLines))
    # angle3 = np.zeros(len(flyLines))

    print('guessing angles')
    print(flyLines.shape)
    print(flyEllipses.shape)
    for ii in (np.arange(flyLines.shape[0]-1)+1):
        dirGuess = angle1[ii-1];

        
        lineAngle = np.rad2deg(np.arctan(flyLines[ii,flynum,0]/flyLines[ii,flynum,1]))+90;    # don't add +90?
        if not np.isnan(lineAngle):
            newDir = degreeVec + lineAngle;
            dirDiff = np.abs(newDir - dirGuess);
            dirDiff[dirDiff > 180] = 360 - dirDiff[dirDiff > 180];
            choice = np.nonzero(np.abs(dirDiff) == np.min(np.abs(dirDiff)));
            try:
                angle1[ii] = lineAngle + degreeVec[choice[0][0]];
            except:
                print((flyLines[ii,flynum,0],flyLines[ii,flynum,1]))
                exit()
        else:
            angle1[ii] = 0.0;

        # dirGuess = angle2[ii-1];
        # newDir = degreeVec + flyEllipses[ii,0,4];
        # dirDiff = abs(newDir - dirGuess);
        # dirDiff[dirDiff > 180] = 360 - dirDiff[dirDiff > 180];
        # choice = np.nonzero(np.abs(dirDiff) == np.min(np.abs(dirDiff)));
        # dirDiff1 = dirDiff[choice[0]];
        # angle2[ii] = flyEllipses[ii,flynum,4] + degreeVec[choice[0][0]]

        # dirGuess = angle3[ii-1];
        # newDir = degreeVec + reducedFlyEllipses[ii,0,4];
        # dirDiff = abs(newDir - dirGuess);
        # dirDiff[dirDiff > 180] = 360 - dirDiff[dirDiff > 180];
        # choice = np.nonzero(np.abs(dirDiff) == np.min(np.abs(dirDiff)));
        # dirDiff1 = dirDiff[choice[0]];
        # angle3[ii] = reducedFlyEllipses[ii,flynum,4] + degreeVec[choice[0][0]];


    angle1 = np.mod(angle1,360);
    # angle2 = np.mod(angle2,360);
    # angle3 = np.mod(angle3,360);

    ##

    # unravel
    angles = angle1;
    dA = np.diff(angles);
    offset = 0;

    outAngles = np.zeros(len(angles));
    outAngles[0] = angles[0];
    for ii in range(len(angles)-1):
        if (dA[ii] > 95):
            offset = offset - 360;
        elif (dA[ii] < -95):
            offset = offset + 360;

        outAngles[ii+1] = angles[ii+1] + offset;


    # check for flips
    # historyvec = range(60)
    # forwardvec = range(30)
    historylen = 60
    forwardlen = 30

    mse = np.zeros(len(outAngles))
    msePlus180 = np.zeros(len(outAngles))
    mseMinus180 = np.zeros(len(outAngles))
    # print(len(outAngles))

    for ii in range(historylen,len(outAngles)-forwardlen):
        fobj = np.polyfit(range(historylen),outAngles[ii-historylen:ii],2)
        mse[ii] = np.mean((outAngles[ii:ii+forwardlen] - (fobj[0]*(historylen+np.arange(forwardlen)+1)**2 + fobj[1]*(historylen+np.arange(forwardlen)+1) + fobj[2]))**2)
        msePlus180[ii] = np.mean((outAngles[ii:ii+forwardlen] + 180 - (fobj[0]*(historylen+np.arange(forwardlen)+1)**2 + fobj[1]*(historylen+np.arange(forwardlen)+1) + fobj[2]))**2)
        mseMinus180[ii] = np.mean((outAngles[ii:ii+forwardlen] - 180 - (fobj[0]*(historylen+np.arange(forwardlen)+1)**2 + fobj[1]*(historylen+np.arange(forwardlen)+1) + fobj[2]))**2)



    MSEthresh = 1.8e5
    # MSEthresh = 1.0e5
    # MSEthresh = 1.0e4
    switches = np.nonzero(mse > MSEthresh)[0];
    # print(np.max(mse))
    # print(switches)
    # print(len(switches))
    st = np.nonzero(np.diff(switches) > 1)[0];
    # st2 = [[1;st+1],[st;length(switches)]];
    stInit = np.insert(st,0,1)
    stFinish = np.append(st,len(switches))
    totaloffset = np.zeros(outAngles.shape[0]);
    # print(switches.shape)
    # print(st)
    # print(stInit)
    # print(stFinish)
    for ii in range(len(stFinish)):
        # print((switches[stInit[ii]],switches[stFinish[ii]]))
        if np.mean(mse[switches[stInit[ii]:stFinish[ii]]]) > np.mean(msePlus180[switches[stInit[ii]:stFinish[ii]]]):
            totaloffset[switches[stInit[ii]]] = 180;
        elif np.mean(mse[switches[stInit[ii]:stFinish[ii]]]) > np.mean(mseMinus180[switches[stInit[ii]:stFinish[ii]]]):
            totaloffset[switches[stInit[ii]]] = -180;

    fixedAngles[flynum] = np.mod(outAngles + np.cumsum(totaloffset),360)

sio.savemat(filename[:-10] + 'fixedAngles.mat',{'automatedAngles':fixedAngles})


angleData = sio.loadmat(filename[:-10] + 'fixedAngles.mat')
fixedAngles = angleData['automatedAngles']


boxSize = 25

modelname = 'Z:/FlyTracker/Tools/flyIdentifiernew_gray__size50000__adam__classic_1.0_1.0_1.0__autoSave.h5'
modelname = '/jukebox/murthy/FlyTracker/Tools/flyIdentifiernew_gray__size50000__adam__classic_1.0_1.0_1.0__autoSave.h5'
# should probably allow the filename and arena type to be command-line defined but until then....
# videoname = '/jukebox/murthy/Dudi/Behavior/DSX_CTRL/150330_0947/150330_0947.avi'
# videoname = '/jukebox/fred/closedLoop/Processed/20170422-170316_male203_vPR6/20170422-170316_male203_vPR6_vid.avi'
videoname = glob.glob(dirName + '/*.avi')[0]
videoname = videoname.replace('\\','/')

# datname = videoname[:-4] + '_tracks.mat'
arena = '16mic'

# load tracks
dat = loadmat(filename)

numframes = fixedAngles.shape[1]
numframes = 100
maleframes = np.zeros((numframes,boxSize*2,boxSize*2,1))
femaleframes = np.zeros((numframes,boxSize*2,boxSize*2,1))

print(videoname)
vr = cv2.VideoCapture(videoname)
vr.set(cv2.cv.CV_CAP_PROP_FOURCC, cv2.cv.CV_FOURCC('H', '2', '6', '4'))

print('loading frames')

for frameInd in xrange(numframes):
	ret, frame = vr.read()
	print(frame.shape)
    frame = np.double(frame,cv2.COLOR_RGB2BGR)
    
# videogen = skvideo.io.vreader(videoname)
# for frameInd,frame in enumerate(videogen):
# 	if frameInd >= dat['pxCenters'].shape[0] or frameInd >= numframes:
# 		break

# 	frame = np.double(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

	# grab tracked fly position
	fx = np.round(dat['pxCenters'][frameInd+1,0,0] + dat['arenaCoords'][0,1]).astype('int')
	fy = np.round(dat['pxCenters'][frameInd+1,0,1] + dat['arenaCoords'][0,0]).astype('int')
	img = rgb2gray(frame[(fx-boxSize):(fx+boxSize),(fy-boxSize):(fy+boxSize),:])
	# print(img.shape)
	M = cv2.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2),-fixedAngles[0,frameInd] + 90,1)
	# print(img.shape)
	# print(cv2.warpAffine(img,M,(img.shape[0],img.shape[1])).shape)
	maleframes[frameInd,:,:,:] = np.expand_dims(cv2.warpAffine(img,M,(img.shape[0],img.shape[1])),axis=-1)

	fx = np.round(dat['pxCenters'][frameInd+1,1,0] + dat['arenaCoords'][0,1]).astype('int')
	fy = np.round(dat['pxCenters'][frameInd+1,1,1] + dat['arenaCoords'][0,0]).astype('int')
	img = rgb2gray(frame[(fx-boxSize):(fx+boxSize),(fy-boxSize):(fy+boxSize),:])
	M = cv2.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2),-fixedAngles[1,frameInd] + 90,1)
	femaleframes[frameInd,:,:,:] = np.expand_dims(cv2.warpAffine(img,M,(img.shape[0],img.shape[1])),axis=-1)

maleframes = maleframes / np.max(maleframes) * 2 - 1
femaleframes = femaleframes / np.max(femaleframes) * 2 - 1

# model = load_model('res/flyOrienterAdam_autoSave.h5')
directionPreds = np.zeros((2,maleframes.shape[0]))
print('loading model')
model = load_model(modelname)
print('predicting fly one frames')
directionPreds[0,:] = np.squeeze(model.predict(maleframes))
print('predicting fly two frames')
directionPreds[1,:] = np.squeeze(model.predict(femaleframes))

# now save them

sio.savemat(filename[:-10] + 'fixedAngles.mat',{'automatedAngles':fixedAngles,'directionGuess':directionPreds})