import json
import random
import matplotlib.pyplot as plt
import numpy as np

from data import data_processing, load_data
import image_processing
import utils
from model import build_model


# Load config
with open("config.json", "r") as config_file:
    config = json.load(config_file)
data_folder_path = config["data_folder_path"]

# Loading and preprocessing data
# More specifically, images undergo different preprocessing steps such as thresholding etc
# Images are also normalized so that pixel values are between 0 and 1
images, labels = load_data.load_images(data_folder_path)
images = image_processing.process_and_save_images(images) / 255.0

# Characters are extracted from the labels and a dictionary is created to map characters to indices
# The labels are then one-hot encoded
# One-hot encoding is done so that the model can understand the data
characters = list(set(''.join(labels)))
char_dict = data_processing.char_dict(characters)

# Call function that uses data to create X and Y arrays for training and testing
# The data is split into training and testing sets
# The training and testing sets are then reshaped
X_train, X_test, y_train, y_test = data_processing.preprocess_data(images, labels, char_dict)
input_shape = (50, 200, 1)
len_symbols = 36
model = build_model(input_shape, len_symbols)
model.summary()
history = utils.train_model(model, X_train, y_train)

# Evaluation
score = model.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]], verbose=1)
print('Test loss and accuracy:', score)

plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)

for i in range(5):
    train_acc_key = f'dense_{2 * i + 1}_accuracy'
    val_acc_key = f'val_dense_{2 * i + 1}_accuracy'

    if train_acc_key in history.history:
        plt.plot(history.history[train_acc_key], label=f'Train Accuracy Layer {i + 1}')
    elif val_acc_key in history.history:
        plt.plot(history.history[val_acc_key], label=f'Validation Accuracy Layer {i + 1}')

plt.title('Model accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Flat predictions are made by taking the argmax of the predictions along the second axis
# The true labels are reshaped to have the number of samples first
# This is done so that the confusion matrix can be plotted
# since the confusion matrix requires 1D arrays
predictions = np.array(model.predict(X_test))
flat_predictions = np.argmax(predictions, axis=2).reshape(-1)
flat_true_labels = np.argmax(y_test, axis=2).reshape(-1)

# Confusion matrix
conf_matrix = utils.plot_confusion_matrix(flat_true_labels, flat_predictions, characters)

# Random tests to see how the model performs, visually
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
random_list = [random.randint(0, len(X_test) - 1) for _ in range(4)]

for i, index in enumerate(random_list):
    axs[i // 2, i % 2].imshow(X_test[index][:, :, 0], cmap='gray')
    axs[i // 2, i % 2].set_title(f"Prediction: {utils.make_prediction(X_test[index], characters, model)}")

plt.show()
