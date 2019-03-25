#

def rgb2bgr(frame):
	return frame[:,:,::-1]

def detectBlinkState(box):
	return np.sum(box);