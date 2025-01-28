import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

class Residual(layers.Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def call(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    # Patch extraction
    layers_list = [
        layers.Conv2D(dim, kernel_size=patch_size, strides=patch_size, padding="valid"),
        layers.Activation("gelu"),
        layers.BatchNormalization()
    ]

    # ConvMixer blocks
    for _ in range(depth):
        layers_list.append(Sequential([
            Residual(Sequential([
                layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same"),
                layers.Activation("gelu"),
                layers.BatchNormalization()
            ])),
            layers.Conv2D(dim, kernel_size=1, padding="valid"),
            layers.Activation("gelu"),
            layers.BatchNormalization()
        ]))

    # Pooling and classification head
    layers_list.extend([
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(n_classes, activation="softmax")
    ])

    # Build the model
    return Sequential(layers_list)