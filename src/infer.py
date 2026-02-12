import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image


def load_model(path):
    return keras.models.load_model(path)


def predict_image(model, image_path, img_size=(224,224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)[0][0]
    return float(pred)


if __name__ == '__main__':
    import sys
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    model = load_model(model_path)
    p = predict_image(model, img_path)
    print('Pneumonia probability:', p)
