import torch as pt
from torch.nn import Sequential, Conv2d, MaxPool2d, BatchNorm2d, ReLU
import pandas as pd
from sklearn.model_selection import train_test_split


class Net():
    def __init__(self):
        super(encoder, self).__init__())

        self.cnn_layers = Sequential(
        # Defining a 2D convolution layer
        Conv2d(1, 2, kernel_size=3, padding=1),
        BatchNorm2d(2),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=2, stride=2),
        # Defining another 2D convolution layer
        Conv2d(2, 2, kernel_size=3, padding=1),
        BatchNorm2d(2),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=2, stride=2),

    )

    def forward(self, model):
        model = self.cnn_layers(model)
        return model
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    """



if __name__ == "__main__":

    #split into train-test
    data_folder = "../Colorization/L_landscapes" # change this to the tensors later
    X_train, X_test, y_train, y_test = train_test_split(data_folder + '/L', data_folder + '/a', test_size = 0.33, random_state = 42)
    epochs = 10
    latent_space_dimn = 20