import numpy as np
# HACK MONKEYPATCH
try:
    import tensorflow as tf
    tf.python.control_flow_ops = tf
except:
    pass
# HACK MONKEYPATCH
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, Conv2DTranspose, Conv3D, MaxPooling2D, UpSampling2D
from keras.layers import LSTM, ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


from helper import *
import keras.losses
# keras.losses.mean_squared_error = mean_squared_error
# from keras.losses import mean_squared_error
import h5py
import glob

import cv2

import argparse
from skimage.color import rgb2gray

from scipy.io import loadmat, savemat
import copy


def ImageAugmentor(alldata,allresp):
	data = copy.copy(alldata)
	resp = copy.copy(allresp)
	pContrast = np.random.random((data.shape[0])) < .3
	pNoise = np.random.random((data.shape[0])) < .1
	pLight = np.random.random((data.shape[0])) < .2
	pFlipX = np.random.random((data.shape[0])) < .5
	pFlipY = np.random.random((data.shape[0])) < .5

	for ii in range(data.shape[0]):
		if pContrast[ii]:
			if np.random.random(1) < 0.5:
				# contrast down
				data[ii] /= np.random.random(1)*3 + 1
			else:
				# contrast up
				data[ii] *= np.random.random(1)*3 + 1

		if pLight[ii]:
			data[ii] += np.random.random(1)*100 - 50
			# for jj in range(data.shape[-1]):
			# 	data[ii,:,:,jj] += np.random.random(1)*100 - 50

		if pNoise[ii]:
			data[ii] += np.random.normal(size=data[ii].shape)*15

		if pFlipX[ii]:
			data[ii] = data[ii,::-1,:,:]
			resp[ii] = resp[ii,::-1,:]

		if pFlipY[ii]:
			data[ii] = data[ii,:,::-1,:]
			resp[ii] = resp[ii,:,::-1]

	# contrast
	# noise
	return np.clip(data,0,255),resp


parser = argparse.ArgumentParser()
parser.add_argument('-gray',dest='gray',action='store_true')
parser.add_argument('-dropout',dest='dropout',action='store_true')
parser.add_argument('-pooling',dest='pooling',action='store_true')
parser.add_argument('-numlayers',dest='numlayers',type=int,default=3)
parser.add_argument('-model',dest='model_name',type=str,default='hourglass')

parser.add_argument('-zscore',dest='zscore',action='store_true')
parser.add_argument('-batchfirst',dest='batchfirst',action='store_true')
parser.add_argument('-augment',dest='manualaugmentation',action='store_true')

parser.add_argument('-load',dest='filename',type=str,default='')
parser.add_argument('-datasize',dest='datasize',type=int,default=-1)
parser.add_argument('-epochs',dest='epochs',type=int,default=100)
parser.add_argument('-convratio',dest='convratio',type=float,default=1.0)
parser.add_argument('-sgd',dest='SGDoption',type=int,default=0)

parser.add_argument('-zoom',dest='zoom',type=float,default=1.0)

# should be ALL or 16mic right now
parser.add_argument('-dataName',dest='dataName',type=str,default='')

pargs = parser.parse_args()
if pargs.filename != '':
	pargs.model_name = 'loadmodel'

outname = 'new'

print('yay')


if pargs.datasize > -1:
    outname += '_size' + str(pargs.datasize) + '_'

if pargs.zscore:
	outname += '_zscore_'
if pargs.batchfirst:
	outname += '_batchfirst_'

if pargs.dataName == '':
	outname += '_adam_' + str(pargs.zoom)
	filename = 'dat/CLframes_' + str(pargs.zoom) + '.mat'
else:
	outname += '_' + pargs.dataName + '_adam_' + str(pargs.zoom)
	filename = 'dat/CLframes_' + pargs.dataName + '_' + str(pargs.zoom) + '.mat'

print('loading ' + filename)
f = h5py.File(filename, 'r')

alldata = np.clip(f['inFrames'][()]*255,0,255)
allresp = f['outFrames'][()]

alldata = np.moveaxis(alldata,[0,1,2,3],[3,2,1,0])
allresp = np.expand_dims(np.moveaxis(allresp,[0,1,2],[2,1,0]),axis=-1)

print(np.max(alldata))
print(np.max(allresp))
if pargs.manualaugmentation:
	augmenteddata,augmentedresp = ImageAugmentor(alldata,allresp)
	alldata = np.concatenate((alldata,augmenteddata))
	allresp = np.concatenate((allresp,augmentedresp))
	outname += '_augmentor_'
print(np.max(alldata))
print(np.max(allresp))

print(alldata.shape)
if pargs.gray:
	outname += '_gray_'
	alldata = alldata[:,:,:,1]
	alldata = np.expand_dims(alldata,axis=-1)
print(alldata.shape)
# exit()

datapts = alldata.shape[0]

print(alldata.shape)
# load fixed xval data
# with h5py.File('dat/tripletDataFix_perm.mat', 'r') as f:
# with h5py.File('dat/cleanTripletData_perm.mat', 'r') as f:
    # test_idx = f['flyFixIdx'][:].astype(np.uintp)
# test_idx = range(100,50000, 1000)
test_idx = np.linspace(10, datapts-1, int(datapts*.1), dtype=int)
# split into training and test set
data_test = alldata[test_idx]
resp_test = allresp[test_idx]
data_train = np.delete(alldata, test_idx, axis=0)
resp_train = np.delete(allresp, test_idx, axis=0)

# set up model
batch_size = 32
nb_epoch = pargs.epochs
nb_classes = allresp.shape[1]

# add in a bunch of dropout in here!!
# https://keras.io/layers/core/#dropout
# model.add(Dropout(0.2))
# also our final layer right now is 3x3x32 (!!) so maybe we want to reduce the amount of pooling
# and also play around with the number of convolutions (>> 32?)

# model names: 'loadmodel', 'classic', 'dropout', 'lesspooling', 'regularization'

if pargs.model_name == 'loadmodel':
	from keras.models import load_model
	outname = pargs.filename[15:-4] + '_p1_'
	print(outname)

	model = load_model(pargs.filename)

elif pargs.model_name == 'hourglass':
	outname += '_hourglass_'
	outname += str(pargs.convratio) + '_' + str(pargs.numlayers) + '_' + str(pargs.pooling) + '_'

	model = Sequential()
	if pargs.batchfirst:
		model.add(BatchNormalization(input_shape=(None,None,1)))
		model.add(Conv2D(int(32*pargs.convratio), (5, 5), padding='valid'))
	else:
		model.add(Conv2D(int(32*pargs.convratio), (5, 5), padding='valid', input_shape=(None,None,alldata.shape[3])))
	# model.add(Conv2D(int(32*pargs.convratio), (5, 5), padding='valid', input_shape=(50,50,1)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	# if pargs.pooling:
	# 	model.add(MaxPooling2D(pool_size=(2, 2)))

	for _ in range(pargs.numlayers-1):	
		if pargs.pooling:
			model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(int(32*pargs.convratio), (5, 5)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

	for _ in range(pargs.numlayers-1):
		if pargs.pooling:
			model.add(Conv2DTranspose(int(32*pargs.convratio), (5, 5), strides=1))
		else:
			model.add(Conv2DTranspose(int(32*pargs.convratio), (5, 5)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		if pargs.pooling:
			model.add(UpSampling2D(size=(2,2)))

	if pargs.pooling:
		model.add(Conv2DTranspose(1, (5, 5), strides=1))
	else:
		model.add(Conv2DTranspose(1, (5, 5)))
	model.add(BatchNormalization())
	model.add(Activation('sigmoid'))

	if pargs.SGDoption == 1:
		outname += '_SGD' + str(pargs.SGDoption) + '_'
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mean_squared_error', optimizer='sgd')
	elif pargs.SGDoption == 2:
		outname += '_SGD' + str(pargs.SGDoption) + '_'
		sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
		model.compile(loss='mean_squared_error', optimizer='sgd')
	elif pargs.SGDoption == 0:
		model.compile(loss='mean_squared_error', optimizer='adadelta')


elif pargs.model_name == 'hourglass_same':
	outname += '_hourglass_same_'
	outname += str(pargs.convratio) + '_' + str(pargs.numlayers) + '_' + str(pargs.pooling) + '_'

	model = Sequential()
	if pargs.batchfirst:
		model.add(BatchNormalization(input_shape=(None,None,1)))
		model.add(Conv2D(int(32*pargs.convratio), (5, 5), padding='same'))
	else:
		model.add(Conv2D(int(32*pargs.convratio), (5, 5), padding='same', input_shape=(None,None,alldata.shape[3])))
	# model.add(Conv2D(int(32*pargs.convratio), (5, 5), padding='valid', input_shape=(50,50,1)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	# if pargs.pooling:
	# 	model.add(MaxPooling2D(pool_size=(2, 2)))

	for _ in range(pargs.numlayers-2):
		model.add(Conv2D(int(32*pargs.convratio), (5, 5), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

	model.add(Conv2D(1, (5, 5), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('sigmoid'))

	if pargs.SGDoption == 1:
		outname += '_SGD' + str(pargs.SGDoption) + '_'
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mean_squared_error', optimizer='sgd')
	elif pargs.SGDoption == 2:
		outname += '_SGD' + str(pargs.SGDoption) + '_'
		sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
		model.compile(loss='mean_squared_error', optimizer='sgd')
	elif pargs.SGDoption == 0:
		model.compile(loss='mean_squared_error', optimizer='adadelta')


elif pargs.model_name == 'hourglass3pool':
	outname += '_hourglass3pool_'
	outname += str(pargs.convratio) + '_' + str(pargs.numlayers) + '_' + str(pargs.pooling) + '_'

	model = Sequential()
	# model.add(Conv2D(int(32*pargs.convratio), (5, 5), padding='valid', input_shape=(None,None,1)))
	model.add(Conv2D(int(32*pargs.convratio), (5, 5), padding='same', input_shape=(None,None,1)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(int(32*pargs.convratio), (4, 4), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(int(32*pargs.convratio), (5, 5), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2DTranspose(int(32*pargs.convratio), (5, 5), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(UpSampling2D(size=(2,2)))

	model.add(Conv2DTranspose(int(32*pargs.convratio), (4, 4), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(UpSampling2D(size=(2,2)))

	model.add(Conv2DTranspose(1, (5, 5), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('sigmoid'))

	if pargs.SGDoption == 1:
		outname += '_SGD' + str(pargs.SGDoption) + '_'
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='mean_squared_error', optimizer='sgd')
	elif pargs.SGDoption == 2:
		outname += '_SGD' + str(pargs.SGDoption) + '_'
		sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
		model.compile(loss='mean_squared_error', optimizer='sgd')
	elif pargs.SGDoption == 0:
		model.compile(loss='mean_squared_error', optimizer='adadelta')


elif pargs.model_name == 'hourglassLSTM':
	outname += '_hourglassLSTM_'
	outname += str(pargs.convratio) + '_' + str(pargs.numlayers) + '_' + str(pargs.pooling) + '_'

	model = Sequential()
	model.add(ConvLSTM2D(int(32*pargs.convratio), (5, 5), padding='valid', data_format='channels_last'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	if pargs.pooling:
		model.add(MaxPooling2D(pool_size=(2, 2)))

	for _ in range(pargs.numlayers-1):	
		model.add(Conv2D(int(32*pargs.convratio), (5, 5)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		if pargs.pooling:
			model.add(MaxPooling2D(pool_size=(2, 2)))

	for _ in range(pargs.numlayers-1):
		if pargs.pooling:
			model.add(Conv2DTranspose(int(32*pargs.convratio), (5, 5), strides=2))
		else:
			model.add(Conv2DTranspose(int(32*pargs.convratio), (5, 5)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

	if pargs.pooling:
		model.add(Conv2DTranspose(1, (5, 5), strides=2))
	else:
		model.add(Conv2DTranspose(1, (5, 5)))
	model.add(BatchNormalization())
	model.add(Activation('sigmoid'))

	if pargs.SGDoption == 1:
		outname += '_SGD' + str(pargs.SGDoption) + '_'
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
	elif pargs.SGDoption == 2:
		outname += '_SGD' + str(pargs.SGDoption) + '_'
		sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
		model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
	elif pargs.SGDoption == 0:
		model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta')



elif pargs.model_name == 'hourglass3D':
	outname += '_hourglass3D_'
	outname += str(pargs.convratio) + '_' + str(pargs.numlayers) + '_' + str(pargs.pooling) + '_'

	model = Sequential()
	model.add(Conv3D(int(32*pargs.convratio), (5, 5), padding='valid', data_format='channels_last'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	if pargs.pooling:
		model.add(MaxPooling2D(pool_size=(2, 2)))

	for _ in range(pargs.numlayers-1):	
		model.add(Conv2D(int(32*pargs.convratio), (5, 5)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		if pargs.pooling:
			model.add(MaxPooling2D(pool_size=(2, 2)))

	for _ in range(pargs.numlayers-1):
		if pargs.pooling:
			model.add(Conv2DTranspose(int(32*pargs.convratio), (5, 5), strides=2))
		else:
			model.add(Conv2DTranspose(int(32*pargs.convratio), (5, 5)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

	if pargs.pooling:
		model.add(Conv2DTranspose(1, (5, 5), strides=2))
	else:
		model.add(Conv2DTranspose(1, (5, 5)))

	model.add(BatchNormalization())
	model.add(Activation('sigmoid'))

	if pargs.SGDoption == 1:
		outname += '_SGD' + str(pargs.SGDoption) + '_'
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer='sgd')
	elif pargs.SGDoption == 2:
		outname += '_SGD' + str(pargs.SGDoption) + '_'
		sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
		model.compile(loss='categorical_crossentropy', optimizer='sgd')
	elif pargs.SGDoption == 0:
		model.compile(loss='categorical_crossentropy', optimizer='adadelta')








# model.compile(loss=mean_squared_error, optimizer='adadelta')
print(outname)
model.summary()

callback = ModelCheckpoint('res/flyPixels' + outname + '_autoSave.h5', verbose=0, monitor='val_loss', save_best_only=True)

# if pargs.manualaugmentation:
# 	datagen = ImageDataGenerator(preprocessing_function=ImageAugmentor)
# 	datagen.fit(data_train,resp_train)

# 	validgen = ImageDataGenerator(preprocessing_function=ImageAugmentor)
# 	validgen.fit(data_test,resp_test)

# 	history = model.fit_generator(generator=datagen.flow(data_train, data_test, batch_size=batch_size),
# 									validation_data=datagen.flow(data_test, resp_test),
# 									steps_per_epoch=len(data_train) / batch_size, epochs=nb_epoch,
# 									verbose=2, callbacks=[callback])
# else:
history = model.fit(data_train, resp_train, batch_size=batch_size, epochs=nb_epoch,
					verbose=2, validation_data=(data_test, resp_test),
					callbacks=[callback])

save_model(model, 'res/flyPixels2' + outname + '')

pred = model.predict(data_test)
with h5py.File('res/flyPixels2' + outname + '_predtest.h5', 'w') as f:
    f.create_dataset("pred", data=pred, compression="gzip")
    f.create_dataset("resp", data=resp_test, compression="gzip")
    f.create_dataset("loss", data=history.history['loss'], compression="gzip")
    f.create_dataset("val_loss", data=history.history['val_loss'], compression="gzip")

pred = model.predict(alldata)
with h5py.File('res/flyPixels2' + outname + '_pred.h5', 'w') as f:
    f.create_dataset("pred", data=pred, compression="gzip")
    f.create_dataset("resp", data=allresp, compression="gzip")
    f.create_dataset("loss", data=history.history['loss'], compression="gzip")
    f.create_dataset("val_loss", data=history.history['val_loss'], compression="gzip")
