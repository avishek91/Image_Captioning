
import tensorflow as tf
import numpy as np
import os

def load_image(image_path, target_size=(299, 299)):
    # Load image from file
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    # Convert image to array
    img = tf.keras.preprocessing.image.img_to_array(img)
    # Preprocess for model, e.g., InceptionV3
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def load_all_images(directory):
    images = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_id = filename.split('.')[0]
            img_path = os.path.join(directory, filename)
            images[image_id] = load_image(img_path)
    return images

def load_captions(filepath):
    captions = {}
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            # Split line into image_id and caption
            # print("line is ",line)
            image_id, caption = line.split(',',1)
            image_id = image_id.split('#')[0]  # Remove #0, #1, etc.
            if image_id not in captions:
                captions[image_id] = []
            # Append the caption text
            captions[image_id].append(caption)
    return captions

def load_data(images_dir, captions_file):
    images = load_all_images(images_dir)
    captions = load_captions(captions_file)
    return images, captions