import cv2
import numpy as np
import os

def process_image_refined(image_path, output_dir, expected_letters=8):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    alpha = 2.0  # Contrast control (1.0-3.0)
    beta = 0    # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Apply binary thresholding
    _, binary = cv2.threshold(adjusted, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by y position first, then by x position to achieve row-wise order
    contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1] // 50, cv2.boundingRect(ctr)[0]))

    # Filter contours based on size and keep only the largest expected number of letters
    filtered_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:expected_letters]

    # Re-sort filtered contours in row-wise order after filtering
    filtered_contours = sorted(filtered_contours, key=lambda ctr: (cv2.boundingRect(ctr)[1] // 50, cv2.boundingRect(ctr)[0]))

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through filtered contours and save individual letters
    saved_letters = []
    for i, contour in enumerate(filtered_contours):
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the letter from the original image using the bounding box
        letter = image[y:y+h, x:x+w]

        # Save the letter image as a PNG file
        letter_filename = os.path.join(output_dir, f'letter_{chr(65 + i)}.png')
        cv2.imwrite(letter_filename, letter)
        saved_letters.append(letter_filename)

    return saved_letters

# Example usage
input_image_path = 'text-extract-8-letters.jpg'
output_directory = 'output_letters'

# Run the function and get the list of saved letter file paths
refined_letter_files = process_image_refined(input_image_path, output_directory)
print("Saved letter files:", refined_letter_files)
