import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import tensorflow as tf
import tf_keras
import tensorflow_hub as hub
from PIL import Image
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of an image using a saved trained model.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("model_path", type=str, help="Path to the .h5 model file")
    parser.add_argument("--category_names", type=str, default=None, help="Path to json file with flowers names")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top predictions")

    args = parser.parse_args()

    reloaded_h5_model = tf_keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    probs, classes = predict(args.image_path, reloaded_h5_model, args.top_k)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        flower_names = [class_names.get(top_class) for top_class in classes]
        print("Flower Names:", flower_names)

    print("Probabilities:", probs)
    print("Classes:", classes)

    
#!python predict.py /content/hard-leaved_pocket_orchid.jpg /content/image_model_1730551280.h5 --category_names /content/label_map.json --top_k 5
