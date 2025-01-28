import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

class Residual(layers.Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def call(self, x):
        return self.fn(x) + x

class DynamicDepthwiseConv2D(layers.Layer):    # Define the dynamic layers
    def __init__(self, kernel_sizes, **kwargs):
        super(DynamicDepthwiseConv2D, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.depthwise_convs = [layers.DepthwiseConv2D(k, padding='same', **kwargs) for k in kernel_sizes]
        self.alpha = self.add_weight(shape=(len(kernel_sizes),),
                                     initializer='ones',
                                     trainable=True,
                                     name='alpha')

    def call(self, x):
        outputs = [conv(x) for conv in self.depthwise_convs]
        alphas = tf.nn.softmax(self.alpha)
        # Direct indexing approach:
        combined = tf.add_n([alphas[i] * outputs[i] for i in range(len(self.kernel_sizes))])
        return combined

def ConvMixer(dim, depth, kernel_sizes=[3,7,9], patch_size=7, n_classes=1000):
    # Patch extraction
    layers_list = [
        layers.Conv2D(dim, kernel_size=patch_size, strides=patch_size, padding="valid"),
        layers.Activation("gelu"),
        layers.BatchNormalization()
    ]

    # ConvMixer blocks with DynamicDepthwiseConv2D
    for _ in range(depth):
        layers_list.append(Sequential([
            Residual(Sequential([
                # Replace the single DepthwiseConv2D with our dynamic layer
                DynamicDepthwiseConv2D(kernel_sizes=kernel_sizes),
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

    return Sequential(layers_list)