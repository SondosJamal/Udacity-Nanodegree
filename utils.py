# This is a sample Python script
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import tf_keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import argparse
np.set_printoptions(suppress=True)

image_size = 224
def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    ps = model.predict(processed_image)
    probabilities = ps[0]
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    probs = probabilities[top_indices]
    classes = [str(index) for index in top_indices]
    return probs, classes
