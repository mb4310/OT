import numpy as np

def re_index_image(Image):
	x_pixels,y_pixels = Image.shape
	A = np.empty([x_pixels * y_pixels, 3])
	for i in np.arange(x_pixels * y_pixels):
		x_index = i // y_pixels				
		y_index = i % y_pixels
		A[i,0] = x_index / x_pixels
		A[i,1] = y_index / y_pixels
		A[i,2] = Image[x_index,y_index] / np.amax(Image)
	return A

def reconstruct_image(X, x_pixels,y_pixels,max_int):
	image = np.empty((x_pixels,y_pixels))
	N_x = X.shape[0]
	for i in np.arange(N_x):
		k = i // y_pixels
		j = i % y_pixels
		image[k,j] = X[i,2]*max_int
	return image