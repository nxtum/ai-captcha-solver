import numpy as np
from sklearn.model_selection import train_test_split


def char_dict(characters):
    """
    Mapping the characters to their index
    """
    return {char: idx for idx, char in enumerate(characters)}


def one_hot_encode(captcha_code, char_dict):
    """
    One hot encoding the captcha code
    This is done by creating a numpy array of zeros with the shape of 5 letters each, 36 possible characters
    Iterate through each letter, and check if it is present
    Used so that the model can understand the data
    """
    one_hot_target = np.zeros((5, 36))  # 5 letters each, 36 possible characters
    for a, b in enumerate(captcha_code):
        if b in char_dict:
            one_hot_target[a, char_dict[b]] = 1  # Iterate through each letter, and check if it is present
    return one_hot_target


def preprocess_data(images, labels, char_dict):
    """
    Preprocess images and labels, and split data into training and testing sets.
    """
    X = np.zeros((len(images), 50, 200, 1))
    Y = np.zeros((5, len(images), 36))

    for index, (image, captcha_code) in enumerate(zip(images, labels)):
        processed_image = image
        target_array = one_hot_encode(captcha_code, char_dict)
        X[index] = processed_image.reshape(50, 200, 1)
        Y[:, index, :] = target_array

    Y_reshaped = Y.transpose(1, 0, 2).reshape(len(images), -1)
    X_train, X_test, y_train_reshaped, y_test_reshaped = train_test_split(X, Y_reshaped, test_size=0.1, random_state=42)
    y_train = y_train_reshaped.reshape(-1, 5, 36).transpose(1, 0, 2)
    y_test = y_test_reshaped.reshape(-1, 5, 36).transpose(1, 0, 2)

    return X_train, X_test, y_train, y_test
