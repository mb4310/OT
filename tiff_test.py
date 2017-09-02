import tifffile
im = tifffile.imread('COEA01-GL-17-0367x0274x0265-8b.tif')
print(type(im))
print(im.shape)
