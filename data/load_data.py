import os
import re

import cv2
import numpy as np


def load_images(folder_path):
    """
    Loads images and labels from the folder path.
    All images are read in grayscale and resized to 200x50 pixels.
    Labels are extracted from the filenames by .png extension.
        """
    images = []
    labels = []
    pattern = re.compile(r"[a-z0-9]+")
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                resized_img = cv2.resize(img, (200, 50))
                if resized_img is not None:
                    label = re.findall(pattern, os.path.splitext(filename)[0].lower())
                    if label:
                        label = ''.join(label)
                        images.append(resized_img)
                        labels.append(label)
                    else:
                        print(f"Invalid label in image: {img_path}")
                else:
                    print(f"Error loading image: {img_path}")
            except Exception as e:
                print(f"Error finding image {img_path}: {str(e)}")

    np.save("numpty_arrays/images.npy", images)
    np.save("numpty_arrays/labels.npy", labels)
    return np.array(images), np.array(labels)
