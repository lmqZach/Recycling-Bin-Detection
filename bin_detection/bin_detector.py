'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import label, regionprops
from roipoly import RoiPoly
import os
import matplotlib.pyplot as plt
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat,disk)

class BinDetector():
	def __init__(self,lr = 0.01, iteration = 10000, pretrained = False):

		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		self.learningRate = lr  # learning rate
		self.iterNumber = iteration  # number of iteration
		# 3 channel and 2 classifer: blue or non-blue

		self.w = np.random.randn(2, 3)
		if pretrained: # assign weight value from training weight file (approach #1)
			self.w = np.load('bin_weight.npy')
		else: # Mannually assign value (copied from training) for gradescope check (approach #2)
			self.w = np.array([[ 0.86034344,  0.31841669, -0.33153088], [-0.86034344, -0.31841669,  0.33153088]])

		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is  blue and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach 
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		mask_img = img.reshape(img.shape[0] * img.shape[1], -1)
		scores = np.dot(mask_img, self.w.T)
		probs = self.softmax(scores)

		# get the label, since label begin at 1, +1 here
		mask_img = np.argmax(probs, axis=1)

		mask_img = mask_img.reshape(img.shape[0], img.shape[1])
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom  right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		footprint = disk(10)
		result = opening(img, footprint)
		result = closing(result, footprint)
		result = opening(result, footprint)
		result = closing(result, footprint)
		label_img = label(result)
		regions = regionprops(label_img)
		boxes = []
		for props in regions:
			minr, minc, maxr, maxc = props.bbox
			x1 = minc
			x2 = maxc
			y1 = minr
			y2 = maxr
			area = abs((x1-x2)*(y1-y2))
			if area<2000:
				continue
			if abs(y1-y2)<abs(x1-x2):
				continue
			boxes.append([x1-20,y1-20,x2+20,y2+20])

		# YOUR CODE BEFORE THIS LINE
		################################################################
		
		return boxes

	def generate_train(self):
		folder = "data/training"
		for filename in os.listdir(folder):
			if filename.endswith(".jpg"):
				img = cv2.imread(os.path.join(folder, filename))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				# display the image and use roipoly for labeling
				fig, ax = plt.subplots()
				ax.imshow(img)
				my_roi = RoiPoly(fig=fig, ax=ax, color='r')

				# get the image mask
				mask = my_roi.get_mask(img)

				# display the labeled region and the image mask
				fig, (ax1, ax2) = plt.subplots(1, 2)
				fig.suptitle('%d pixels selected\n' % img[mask, :].shape[0])

				ax1.imshow(img)
				ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
				ax2.imshow(mask)
				np.save(r'data/training/'+str(filename[0:4])+'.npy', mask)
				plt.show(block=True)
		pass

	def softmax(self, X):
		# calculate the softmax
		exp = np.exp(X)
		sum_exp = np.sum(exp, axis=1, keepdims=True)
		softmax = exp / sum_exp
		return softmax

	def accuracy(self, pred, true):
		return sum(true == pred) / len(pred)

	def predict(self, test_data, w):
		scores = np.dot(test_data, w.T)
		probs = self.softmax(scores)
		return np.argmax(probs, axis=1).reshape((-1, 1))

	def train(self):
		w = np.random.randn(2,3)
		for t in range(self.iterNumber):

			train_example = np.random.randint(1,61)
			#load label
			file_y = "data/training/"+str(train_example).zfill(4)+'.npy'
			y = np.load(file_y)
			M = y.shape[0] * y.shape[1]
			y = y.reshape(M)
			y = np.int32(y)
			M = y.shape[0] * y.shape[1]
			true = y.copy()
			temp = np.zeros((M, 2))
			# print(temp)
			temp[np.arange(M), y] = 1
			# print(temp)
			y = temp
			# load data
			file_x = "data/training/" + str(train_example).zfill(4) + '.jpg'
			img = cv2.imread(file_x)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			x = img.reshape(M, -1)
			# print(x.shape)
			y1 = self.softmax(np.dot(x, w.T))
			# print(y1)
			# print(y1)
			loss = - (1.0 / M) * np.sum(y * np.log(y1))
			# loss is for whole dataset loss
			# los in dw is for updating the parameter. It is the loss for each parameter
			dw = -(1.0 / M) * np.dot((y - y1).T, x) + 0.01 * w
			w -= self.learningRate * dw
			pred = self.predict(x, w)
			acc = self.accuracy(pred.ravel(), true)
			print(acc)
		np.save('weight.npy', w)
		pass
