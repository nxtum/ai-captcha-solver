from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.models import Model


def build_model(input_shape, len_symbols):
    """
    This model is a simple CNN that takes an image as input and outputs a prediction for each character in the captcha.
    Filters are used to extract features from the image, and the output is passed through a dense layer for each character.
    Convolutional layers are used to extract features from the image, and max pooling layers are used to reduce the spatial dimensions.
    Batch normalization is used to normalize the activations of the previous layer at each batch.
    Dropout is used to prevent overfitting by randomly setting a fraction of the input units to 0 at each update during training.
    """
    captcha = Input(shape=input_shape)
    x = captcha

    filters = [32, 64, 128, 256]

    for filt in filters:
        x = Conv2D(filt, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    flat_output = Flatten()(x)

    dense_dropout = Dropout(0.5)  # Layer for each character

    output1 = dense_block(flat_output, len_symbols, dense_dropout)
    output2 = dense_block(flat_output, len_symbols, dense_dropout)
    output3 = dense_block(flat_output, len_symbols, dense_dropout)
    output4 = dense_block(flat_output, len_symbols, dense_dropout)
    output5 = dense_block(flat_output, len_symbols, dense_dropout)

    model = Model(inputs=captcha, outputs=[output1, output2, output3, output4, output5])

    return model


def dense_block(input_layer, len_symbols, dense_dropout):
    dense = Dense(64, activation='relu')(input_layer)
    dropout_output = dense_dropout(dense)
    output = Dense(len_symbols, activation='sigmoid')(dropout_output)
    return output
