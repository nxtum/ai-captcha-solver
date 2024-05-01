import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.src.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, symbols):
    """
    Plots confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=symbols, yticklabels=symbols)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def make_prediction(captcha, symbols, model):
    """
    Makes prediction for captcha
    """
    captcha = captcha.reshape(50, 200)
    result = model.predict(captcha.reshape(1, 50, 200, 1))
    result = np.reshape(result, (5, 36))
    indexes = [np.argmax(i) for i in result]
    label = ''.join(symbols[i] for i in indexes)
    return label


def train_model(model, X_train, y_train):
    """
    Trains the model and return training history.
    """
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=7, restore_best_weights=True)
    history = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]],
                        batch_size=32, epochs=50, verbose=1, validation_split=0.2, callbacks=[early_stopping])
    return history
