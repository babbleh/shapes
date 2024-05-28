import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_waveform(length, amplitude=10, frequency=1):
    x = np.linspace(0, 2 * np.pi, length)
    waveform = amplitude * np.cos(frequency * x)
    return waveform

def apply_texture_to_shape(image, mask):
    rows, cols = mask.shape
    textured_image_x = np.zeros_like(mask, dtype=np.float32)
    textured_image_y = np.zeros_like(mask, dtype=np.float32)

    # Apply texture row by row (x direction)
    for row in range(rows):
        in_shape = False
        start_col = 0
        
        for col in range(cols):
            if mask[row, col] == 255 and not in_shape:
                in_shape = True
                start_col = col
            elif (mask[row, col] != 255 and in_shape) or (col == cols - 1 and in_shape):
                in_shape = False
                end_col = col + 1 if col == cols - 1 else col
                length = end_col - start_col
                waveform = generate_waveform(length)
                textured_image_x[row, start_col:end_col] = waveform
    
    # Apply texture column by column (y direction)
    for col in range(cols):
        in_shape = False
        start_row = 0
        
        for row in range(rows):
            if mask[row, col] == 255 and not in_shape:
                in_shape = True
                start_row = row
            elif (mask[row, col] != 255 and in_shape) or (row == rows - 1 and in_shape):
                in_shape = False
                end_row = row + 1 if row == rows - 1 else row
                length = end_row - start_row
                waveform = generate_waveform(length)
                textured_image_y[start_row:end_row, col] = waveform

    # Combine the x and y direction textures
    combined_texture = (textured_image_x + textured_image_y) / 2
    
    output = np.zeros_like(mask, dtype=np.float32)
    output[mask == 255] = combined_texture[mask == 255]
    
    return output

def process_image(input_image):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to isolate black shapes
    _, thresholded = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of the shapes
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    textured_image = np.zeros_like(gray, dtype=np.float32)
    
    for contour in tqdm(contours):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        shape_texture = apply_texture_to_shape(input_image, mask)
        
        # Blend the shape texture into the overall textured image
        textured_image[mask == 255] = shape_texture[mask == 255]
    
    textured_image = cv2.normalize(textured_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge([textured_image] * 3)

# Load input image
input_file_path = '/Users/henrydawson/onlyplants/voronois/1k/seed_775.png'
input_image = cv2.imread(input_file_path)

# Process the image to apply textures to all shapes
output_image = process_image(input_image)

# Display the input and output images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Input Image')
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title('Textured Image')
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.show()
