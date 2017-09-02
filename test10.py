import numpy as np
from numpy.linalg import inv
from scipy import misc
import pandas as pd 
import matplotlib.pyplot as plt

d = 3

image_1 = 'img1.tiff'
image_2 = 'img4.tiff'

i_1 = misc.imread(image_1, flatten=True)
j_1 = misc.imread(image_2, flatten=True)


def re_index_image(Image):
	x_pixels,y_pixels = Image.shape
	A = np.empty([x_pixels * y_pixels, 3])
	for i in np.arange(x_pixels * y_pixels):
		x_index = i // y_pixels
		y_index = i % y_pixels
		A[i,0] = x_index
		A[i,1] = y_index
		A[i,2] = Image[x_index,y_index]
	return A

x = re_index_image(i_1)
y = re_index_image(j_1)

df1 = pd.DataFrame(x)
df2 = pd.DataFrame(y)
print(df2.describe())
print(df1.describe())

plt.hist(df1[2], bins=20)
plt.hist(df2[2], bins=20)
plt.show()

