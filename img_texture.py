import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#help(np.random)

def generate_waveform(length, amplitude=10, frequency=1):
    x = np.linspace(0, 2 * np.pi, length)
    waveform = amplitude * np.cos(frequency * x)
    return waveform

input_file_path = '/Users/henrydawson/onlyplants/voronois/1k/seed_775.png'
input_image = cv2.imread(input_file_path)

gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Threshold the image to isolate black shapes
_, thresholded = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

# Scan the image by reading row by row, from left to right, generate a series of tuples
# by identifying consecutive runs of pixel values so that
# (index of row, index of pixel before start, value of pixel before start, index of pixel after end, value of pixel after end)

ex = np.random.randint(2, size=(30,30))

b = np.pad(ex, 1, "constant", constant_values=0)[1:ex.shape[0]+1,2:ex.shape[1]+2]
print(ex.shape, b.shape)


edges = ex - b
#such that pixel >0: last pixel of the run
# pixel < 1
print(edges[0,9])
print(ex[0,9], ex[0,10], ex[0,19], ex[0,20])
print(np.argwhere(edges != 0))

# maybe a good way to do this is to think of edge detection filters -- I want to find the moment where pixel values differ. 
# If I subtract the matrix from itself from the left (new = a[0:1] - a[0:2]) then
#  I will find one edge with a positive value, and another edge with a negative value
# This assumes that it is faster to scan and: if beginning edge value, sample pixel and begin a run,
#   if 0, continue the run, 
#   and if end edge value, sample the pixel and end the run. 
# than to scan and: if pixel != pixel before , begin run, 


# Find contours of the shapes
#contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#textured_image = np.zeros_like(gray, dtype=np.float32)



# Display the input and output images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Input Image')
plt.imshow(ex)
plt.subplot(1, 2, 2)
plt.title('w/e')
plt.imshow(edges)
plt.show() 