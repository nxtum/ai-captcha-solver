import cv2
import numpy as np



def process_images(images):
    """
    Process the images using adaptive thresholding, Gaussian blur, dilation, etc.
    Adaptive threshold works by calculating the threshold for a small region of the image.
    The threshold value is the mean of the neighborhood area minus a constant value.

    Morporhology tries to remove noise and smooth the image.
    Dilating the image increases the white region in the image.
    Gaussian blur is used to reduce noise and detail in the image.
    """
    processed_images = []
    for idx in range(len(images)):
        original_image = images[idx]

        th = cv2.adaptiveThreshold(original_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 191, 31)
        val, th3 = cv2.threshold(th, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((4, 4), np.uint8)
        close_img1 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
        dilated_img = cv2.dilate(close_img1, np.ones((3, 3), np.uint8), iterations=1)
        gauss_img2 = cv2.GaussianBlur(dilated_img, (3, 3), 0)

        processed_images.append(gauss_img2)

    return np.array(processed_images)


def process_and_save_images(images):
    processed_images = process_images(images)
    np.save("numpty_arrays/processed_images.npy", processed_images)
    return np.array(processed_images)

