import numpy as np
import scipy.io as sio
import sys
import glob
import copy




def findDist(centers1,centers2):
   distances = np.zeros((centers1.shape[0],centers2.shape[0]))
   for ii in list(range(centers1.shape[0])):
      for jj in list(range(centers2.shape[0])):
         distances[ii][jj]=np.sqrt((centers1[ii][0]-centers2[jj][0])**2 + (centers1[ii][1]-centers2[jj][1])**2)

   return distances

def matchFlies(flyCenters, oldCenters, nFlies=2):

   # nFlies = flyTrackerSettings.nFlies
   # flyCenters = flyTrackerSettings.flyCenters
   # oldCenters = flyTrackerSettings.oldCenters

   if np.sum(oldCenters) == 0:
      if (np.sum(flyCenters) != 0):
         oldLabels = np.arange(nFlies)
         newCenters = np.zeros(oldCenters.shape)
         D = findDist(oldCenters,flyCenters);
         c = np.linspace(0,nFlies-1,nFlies).astype(int)

         # print(D,oldCenters,flyCenters)
         pathLabels = m.compute(D)
         newLabels = np.arange(nFlies)
         for ii in list(range(nFlies)):
            newLabels[ii] = pathLabels[ii][1]
            newCenters[newLabels[ii]] = flyCenters[ii]
            oldLabels[ii] = c[ii]

         return newCenters, newLabels, oldLabels

      else:
         return flyCenters, list(range(nFlies)), list(range(nFlies))

   oldLabels = list(range(nFlies))
   newLabels = list(range(nFlies))
   newCenters = np.zeros(oldCenters.shape)
   D = findDist(flyCenters, oldCenters)
   c = np.linspace(0,nFlies-1,nFlies)

   pathLabels = m.compute(D)
   for ii in list(range(nFlies)):
      newLabels[ii] = pathLabels[ii][1]
      newCenters[newLabels[ii]] = flyCenters[ii]
      oldLabels[ii] = c[ii]

   return newCenters, newLabels, oldLabels


def unwrapAngles(inAngles,thresh,maxVal):
   # when angles go from 0-360 or vice versa, unwrap them for interpolation!
   outAngles = np.zeros(inAngles.shape)
   dA = np.diff(inAngles)
   offset = 0;
   outAngles[0] = inAngles[0]
   for ii in range(outAngles.shape[0]-1):
      if (dA[ii] > thresh):
         offset -= maxVal
      elif (dA[ii] < -thresh):
         offset += maxVal

      outAngles[ii+1] = inAngles[ii+1] + offset

   return outAngles


from munkres import Munkres
m = Munkres()


dirName = sys.argv[1]



try:
  filename = glob.glob(dirName + '/*_tracks.mat')[0]
  filename = filename.replace('\\','/')
  globals().update(sio.loadmat(filename))
  print('loaded file')
  centroids = pxCenters

except:
  try:
    print(dirName + '/*_centroids.mat')
    filename = glob.glob(dirName + '/*_centroids.mat')[0]
    filename = filename.replace('\\','/')
    globals().update(sio.loadmat(filename))

    filename = glob.glob(dirName + '/*_annotated2.mat')[0]
    filename = filename.replace('\\','/')
    globals().update(sio.loadmat(filename))
    print('loaded file')

    # fix fly identity

    # for ii,center in enumerate(centroids):
    #   if ii == 0:
    #     newCenters, newLabels, _ = matchFlies(center,np.zeros((2,2)))
    #   else:
    #     newCenters, newLabels, _ = matchFlies(center,centroids[ii-1])

    #   centroids[ii] = newCenters

    #   oldLines = flyLines[ii]
    #   oldEllipses = flyEllipses[ii]
    #   for flyNum in range(2):
    #     flyLines[ii,flyNum] = oldLines[newLabels[flyNum]]
    #     flyEllipses[ii,flyNum] = oldEllipses[newLabels[flyNum]]
  except:
    print('failed to find tracks or centroids')
    exit()

nFlies = centroids.shape[1]

centerX = data['arenaCenter'][0][0]
centerY = data['arenaCenter'][0][1]
radius = data['arenaRadius'][0][0]
arenaCoords = (max(0,centerX-radius), max(0,centerY-radius), min(centerX+radius,width), min(centerY+radius,height));

# first pass, go through and match all flies to be close to each other
for ii,center in enumerate(centroids):
      if ii == 0:
        newCenters, newLabels, _ = matchFlies(center,np.zeros((nFlies,2)),nFlies=nFlies)
      else:
        newCenters, newLabels, _ = matchFlies(center,centroids[ii-1],nFlies=nFlies)

      centroids[ii] = newCenters

      oldLines = flyLines[ii]
      oldEllipses = flyEllipses[ii]
      tmpLines = np.zeros((flyLines.shape[1],flyLines.shape[2]))
      tmpEllipses = np.zeros((flyEllipses.shape[1],flyEllipses.shape[2]))

      # print(newLabels)

      for flyNum in list(range(nFlies)):
        tmpLines[flyNum] = oldLines[newLabels[flyNum]]
        tmpEllipses[flyNum] = oldEllipses[newLabels[flyNum]]

      flyLines[ii] = tmpLines
      flyEllipses[ii] = tmpEllipses



# sio.savemat(filename[:-10] + 'fixedAngles.mat',{'centroids':centroids,'flyLines':flyLines,'oldAngles':oldAngles,'automatedAngles':fixedAngles,'finalAngles':finalAngles})
# sio.savemat(filename[:-10] + 'fixedAngles.mat',{'centroids':centroids,'finalAngles':flyLines})










# # now go through and check when the fly fictively 'leaps' and interpolate the position/angle
# # iterate through this a few times?? how to tell when there is a real leap vs fictive one?
# for rr in list(range(30)):
#   thresh = 20
#   for flynum in list(range(nFlies)):
#     # dx = np.diff(centroids[:,flynum,0],n=rr+1)
#     # dy = np.diff(centroids[:,flynum,1],n=rr+1)
#     dx = np.diff(centroids[:,flynum,0],n=1)
#     dy = np.diff(centroids[:,flynum,1],n=1)

#     # find where it is NOT leaping around
#     # leap = np.where(np.logical_not(np.logical_or(np.abs(dx) > thresh*rr, np.abs(dy) > thresh*rr)))
#     # print(np.where(np.logical_or(np.abs(dx) > thresh, np.abs(dy) > thresh)))
#     # lept = np.where(np.logical_or(np.abs(dx) > thresh*(rr+1), np.abs(dy) > thresh*(rr+1)))
#     lept = np.where(np.logical_or(np.logical_or(np.isnan(dx), np.isnan(dy)),np.logical_or(np.abs(dx) > thresh, np.abs(dy) > thresh)))[0]
#     # print(lept)
#     centroids[lept,flynum,:] = None
#     # leptFrom = lept-1
#     # centroids[lept,flynum,:] = centroids[leptFrom,flynum,:]




# for flynum in list(range(nFlies)):
#   dx = np.diff(centroids[:,flynum,0])
#   dy = np.diff(centroids[:,flynum,1])

#   # find where it is NOT leaping around
#   leap = np.where(np.logical_not(np.logical_or(np.isnan(centroids[:,flynum,0]), np.isnan(centroids[:,flynum,0]))))
#   centroids[:,flynum,0] = np.interp(np.arange(centroids.shape[0]), leap[0], centroids[leap,flynum,0][0])
#   centroids[:,flynum,1] = np.interp(np.arange(centroids.shape[0]), leap[0], centroids[leap,flynum,1][0])
#   # centroids[lept,flynum,0] = 0

#   # circular interpolation, this goes from 0 to 1
#   # is this circular?!
#   flyLines[:,flynum,0] = np.mod(np.interp(np.arange(flyLines.shape[0]), leap[0], unwrapAngles(flyLines[leap,flynum,0][0],0.55,1)),1)
#   # circular interpolation, this goes from -1 to +1
#   flyLines[:,flynum,1] = np.mod(np.interp(np.arange(flyLines.shape[0]), leap[0], unwrapAngles(flyLines[leap,flynum,1][0]+1,1.1,2)),2)-1
#   flyLines[:,flynum,2] = np.interp(np.arange(flyLines.shape[0]), leap[0], flyLines[leap,flynum,2][0])
#   flyLines[:,flynum,3] = np.interp(np.arange(flyLines.shape[0]), leap[0], flyLines[leap,flynum,3][0])

#   for vals in list(range(4)):
#     flyEllipses[:,flynum,vals] = np.interp(np.arange(flyEllipses.shape[0]), leap[0], flyEllipses[leap,flynum,vals][0])

#   # circular interpolation, this goes from 0 to 180
#   flyEllipses[:,flynum,-1] = np.interp(np.arange(flyEllipses.shape[0]), leap[0], flyEllipses[leap,flynum,-1][0], period=180)









# and do this again just in case...
for ii,center in enumerate(centroids):
      if ii == 0:
        newCenters, newLabels, _ = matchFlies(center,np.zeros((nFlies,2)),nFlies=nFlies)
      else:
        newCenters, newLabels, _ = matchFlies(center,centroids[ii-1],nFlies=nFlies)

      centroids[ii] = newCenters

      oldLines = flyLines[ii]
      oldEllipses = flyEllipses[ii]
      tmpLines = np.zeros((flyLines.shape[1],flyLines.shape[2]))
      tmpEllipses = np.zeros((flyEllipses.shape[1],flyEllipses.shape[2]))
      for flyNum in list(range(2)):
        tmpLines[flyNum] = oldLines[newLabels[flyNum]]
        tmpEllipses[flyNum] = oldEllipses[newLabels[flyNum]]

      flyLines[ii] = tmpLines
      flyEllipses[ii] = tmpEllipses


# fixedAngles = np.zeros((2,len(angles)))
fixedAngles = np.zeros((nFlies,len(flyLines)))
oldAngles = np.zeros((nFlies,len(flyLines)))
for flynum in list(range(nFlies)):
    degreeVec = np.arange(-360,540,180);

    angle1 = np.zeros(len(flyLines))

    print('guessing angles')
    print(flyLines.shape)
    print(flyEllipses.shape)
    for ii in (np.arange(flyLines.shape[0]-1)+1):
        dirGuess = angle1[ii-1];

        lineAngle = np.rad2deg(np.arctan(flyLines[ii,flynum,0]/flyLines[ii,flynum,1]))+90;    # don't add +90?
        oldAngles[flynum,ii] = lineAngle
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


    angle1 = np.mod(angle1,360);
    ##

    # unravel
    angles = angle1;
    dA = np.diff(angles);
    offset = 0;

    outAngles = np.zeros(len(angles));
    outAngles[0] = angles[0];
    for ii in list(range(len(angles)-1)):
        if (dA[ii] > 95):
            offset = offset - 360;
        elif (dA[ii] < -95):
            offset = offset + 360;

        outAngles[ii+1] = angles[ii+1] + offset;


    historylen = 60
    forwardlen = 30

    mse = np.zeros(len(outAngles))
    msePlus180 = np.zeros(len(outAngles))
    mseMinus180 = np.zeros(len(outAngles))
    # print(len(outAngles))

    for ii in list(range(historylen,len(outAngles)-forwardlen)):
        fobj = np.polyfit(list(range(historylen)),outAngles[ii-historylen:ii],2)
        mse[ii] = np.mean((outAngles[ii:ii+forwardlen] - (fobj[0]*(historylen+np.arange(forwardlen)+1)**2 + fobj[1]*(historylen+np.arange(forwardlen)+1) + fobj[2]))**2)
        msePlus180[ii] = np.mean((outAngles[ii:ii+forwardlen] + 180 - (fobj[0]*(historylen+np.arange(forwardlen)+1)**2 + fobj[1]*(historylen+np.arange(forwardlen)+1) + fobj[2]))**2)
        mseMinus180[ii] = np.mean((outAngles[ii:ii+forwardlen] - 180 - (fobj[0]*(historylen+np.arange(forwardlen)+1)**2 + fobj[1]*(historylen+np.arange(forwardlen)+1) + fobj[2]))**2)



    MSEthresh = 1.8e5
    MSEthresh = 1.0e5
    switches = np.nonzero(mse > MSEthresh)[0];
    st = np.nonzero(np.diff(switches) > 1)[0];
    stInit = np.insert(st,0,1)
    stFinish = np.append(st,len(switches))
    totaloffset = np.zeros(outAngles.shape[0]);
    for ii in list(range(len(stFinish))):
        if np.mean(mse[switches[stInit[ii]:stFinish[ii]]]) > np.mean(msePlus180[switches[stInit[ii]:stFinish[ii]]]):
            totaloffset[switches[stInit[ii]]] = 180;
        elif np.mean(mse[switches[stInit[ii]:stFinish[ii]]]) > np.mean(mseMinus180[switches[stInit[ii]:stFinish[ii]]]):
            totaloffset[switches[stInit[ii]]] = -180;

    fixedAngles[flynum] = np.mod(outAngles + np.cumsum(totaloffset),360)

sio.savemat(filename[:-10] + 'fixedAngles.mat',{'centroids':centroids,'flyLines':flyLines,'oldAngles':oldAngles,'automatedAngles':fixedAngles,'arenaCoords':arenaCoords})

finalAngles = copy.copy(fixedAngles)



def findAngleFlips(saveAngle,saveFV):

   numRotatingFrames = 10
   rotatedThreshold = 90
   nFlies = saveAngle.shape[0]
   # if a fly moves ~ 90 degrees in a couple frames, REALLY examine that point
   for ff in list(range(nFlies)):
      scarypts = np.where(np.abs(movingsum(np.diff(saveAngle[ff,0:frameCount+1],axis=0),numRotatingFrames)) > rotatedThreshold)[0]
      scarypts = scarypts[scarypts != 0]
      scarypts = np.append(np.zeros(1),scarypts)
      scarypts = np.append(scarypts,frameCount)

      scarystart = np.where(np.diff(scarypts) > numRotatingFrames/2)[0]+1
      scaryend = np.where(np.diff(scarypts) > numRotatingFrames/2)[0]

      if (any(scarypts!=0)):
         for cc in list(range(len(scarystart))):
            if identifyMoonwalkers(saveFV,round((scarypts[scaryend[cc]]+scarypts[scarystart[cc]])/2),scarypts[scarystart[cc]]-scarypts[scaryend[cc]])[ff]:
               saveFV, saveAngle = flipFlies(saveFV,saveAngle,ff,scarypts[scaryend[cc]],scarypts[scarystart[cc]])
               # fixRotation(ff,scarypts[scarystart[cc-1]],scarypts[scaryend[cc]])

   return saveFV, saveAngle

def computeMovement(currFrame,saveCenters,saveAngle):

   nFlies = saveAngle.shape[0]
   FV = np.zeros(nFlies)

   forward = np.zeros(nFlies)
   if (currFrame > 0):
      for ii in list(range(nFlies)):
         # make sure I'm not off by one..
         DV = np.squeeze(np.diff(saveCenters[(currFrame-1):(currFrame+1),ii,:],axis=0))
         mvAngle = np.arctan2(DV[0],DV[1])
         TV = np.sqrt(np.sum(np.float32(DV[:])**2))

         FV[ii] = np.sum(TV * np.cos(np.deg2rad(saveAngle[ii,currFrame]+90 % 360)-mvAngle))

   return FV

def computeMovement2(currFrame=None):
   if currFrame is None:
      currFrame = frameCount

   FV = np.zeros(nFlies)
   LV = np.zeros(nFlies)
   RV = np.zeros(nFlies)

   forward = np.zeros(nFlies)
   if (currFrame > 0):
      for ii in range(nFlies):
         # make sure I'm not off by one..
         DV = np.squeeze(np.diff(saveCenters[(currFrame-1):(currFrame+1),ii,:],axis=0))
         mvAngle = np.arctan2(DV[0],DV[1])
         TV = np.sqrt(np.sum(np.float32(DV[:])**2))

         FV[ii] = np.sum(TV * np.cos(np.deg2rad(saveAngle[currFrame,ii]+90 % 360)-mvAngle))
         LV[ii] = np.sum(TV * np.sin(np.deg2rad(saveAngle[currFrame,ii]+90 % 360)-mvAngle))


         RV[ii] = saveAngle[currFrame,ii] - saveAngle[currFrame-1,ii]
         if (RV[ii] > 180):
            RV[ii] -= 360
         elif (RV[ii] < -180):
            RV[ii] += 360

   return FV,LV,RV


def identifyMoonwalkers(saveFV,targetFrame,numFrameSmoothing):
   moonwalk = np.mean(saveFV[:,int(targetFrame-numFrameSmoothing/2):int(targetFrame+numFrameSmoothing/2+1)],axis=1) < 0;

   return moonwalk

def flipFlies(flyNum,startOrientation,endOrientation):
   print('flipping fly ' + str(flyNum) + ' between ' + str(startOrientation) + ' and ' + str(endOrientation))
   flyNum = int(flyNum)
   startOrientation = int(startOrientation)
   endOrientation = int(endOrientation)
   saveFV[startOrientation:endOrientation+1,flyNum] *= -1;
   saveAngle[startOrientation:endOrientation+1,flyNum] += 180;
   return saveFV,saveAngle


def orientFlies(saveFV,saveAngle,startFrame,frameCount):
   nFlies = saveAngle.shape[0]
   saveFV[:,1] = 0
   for ii in list(range(nFlies)):
      if np.mean(saveFV[ii,startFrame:frameCount]) < 0:
         print('oh dear, fly #' + str(ii) + ' needs some guidance!')
         saveFV[ii,startFrame:] *= -1
         saveAngle[ii,startFrame:] += 180



print('double-checking flies')

angularThresh = 50
loopnum = 1;

saveFV = np.zeros((finalAngles.shape))
for ii in list(range(finalAngles.shape[1])):
  # saveFV[:,ii] = computeMovement(ii,pxCenters,finalAngles)
  saveFV[:,ii] = computeMovement(ii,centroids,finalAngles)

frameStep = 1000
for startFrame in list(range(0,finalAngles.shape[1]-frameStep,frameStep)):
    orientFlies(saveFV,finalAngles,startFrame,startFrame+frameStep)     # we should also be finding jumps and looking at orientations between each one


sio.savemat(filename[:-10] + 'fixedAngles.mat',{'centroids':centroids,'flyLines':flyLines,'oldAngles':oldAngles,'automatedAngles':fixedAngles,'finalAngles':finalAngles,'arenaCoords':arenaCoords})
