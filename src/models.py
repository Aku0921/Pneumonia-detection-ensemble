import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    EfficientNetB0,
    MobileNetV2,
    ResNet50V2,
    DenseNet121,
    VGG16,
)
from tensorflow.keras.applications import (
    efficientnet,
    mobilenet_v2,
    resnet_v2,
    densenet,
    vgg16,
)


BACKBONES = {
    "efficientnetb0": (EfficientNetB0, efficientnet.preprocess_input, 224),
    "mobilenetv2": (MobileNetV2, mobilenet_v2.preprocess_input, 224),
    "resnet50v2": (ResNet50V2, resnet_v2.preprocess_input, 224),
    "densenet121": (DenseNet121, densenet.preprocess_input, 224),
    "vgg16": (VGG16, vgg16.preprocess_input, 224),
}


def create_model(backbone_name="efficientnetb0", input_shape=(224, 224, 3), dropout=0.2, base_trainable=False):
    """Create a binary classification model using the requested backbone.

    Returns (model, base_model)
    """
    backbone_name = backbone_name.lower()
    if backbone_name not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone_name}. Options: {list(BACKBONES.keys())}")

    BaseCls, preprocess_fn, default_size = BACKBONES[backbone_name]

    inputs = keras.Input(shape=input_shape)

    # Basic augmentation in-model so it's consistent between training and export
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip('horizontal')(x)
    x = layers.RandomRotation(0.06)(x)
    x = layers.RandomZoom(0.06)(x)

    # Multiply back to 0-255 for model-specific preprocessing
    x = layers.Rescaling(255.0)(x)

    # instantiate backbone and call it on preprocessed tensor
    base = BaseCls(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = base_trainable

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    return model, base
