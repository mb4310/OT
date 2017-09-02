import numpy as np 
from scipy import misc
import pandas as pd 

image_1 = 'puppy.jpg'
image_2 = 'flower.jpg'

I_1 = np.array(misc.imread(image_1))
I_2 = np.array(misc.imread(image_2))

I_1_new = np.empty((I_1.shape[0],I_1.shape[1]))
for i in np.arange(I_1.shape[0]):
	for j in np.arange(I_1.shape[1]):
		I_1_new[i,j] = I_1[i,j,0]


print(np.amax(I_1_new))

def re_index_image(Image):
	x_pixels,y_pixels = Image.shape
	A = np.empty([x_pixels * y_pixels, 3])
	for i in np.arange(x_pixels * y_pixels):
		x_index = i // y_pixels
		y_index = i % y_pixels
		A[i,0] = x_index / x_pixels
		A[i,1] = y_index / y_pixels
		A[i,2] = Image[x_index,y_index]/np.amax(np.array(Image))
	return A

def reconstruct_image(X, x_pixels,y_pixels,max_intensity):
	image = np.empty((x_pixels,y_pixels))
	N_x = X.shape[0]
	for i in np.arange(N_x):
		k = i // y_pixels
		j = i % y_pixels
		image[k,j] = X[i,2] * max_intensity
	return 









def re_index_image_N(Image):
	x_pixels,y_pixels = Image.shape
	A = np.empty([x_pixels * y_pixels, 3])
	for i in np.arange(x_pixels * y_pixels):
		x_index = i // y_pixels
		y_index = i % y_pixels
		A[i,0] = x_index 
		A[i,1] = y_index 
		A[i,2] = Image[x_index,y_index]
	return A

def reconstruct_image_N(X, x_pixels,y_pixels):
	image = np.empty((x_pixels,y_pixels))
	N_x = X.shape[0]
	for i in np.arange(N_x):
		k = i // y_pixels
		j = i % y_pixels
		image[k,j] = X[i,2] 
	return image