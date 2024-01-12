#!/usr/bin/env python
# coding: utf-8

from urllib import request
from PIL import Image
from io import BytesIO
import tflite_runtime.interpreter as tflite
import numpy as np

target_size = (75, 75)
class_names = ['aquatic_mammals', 
               'fish', 
               'flowers', 
               'food_containers', 
               'fruit_and_vegetables',
               'household_electrical_devices', 
               'household_furniture', 
               'insects', 
               'large_carnivores', 
               'large_man-made_outdoor_things',
               'large_natural_outdoor_scenes', 
               'large_omnivores_and_herbivores', 
               'medium_mammals', 
               'non-insect_invertebrates', 
               'people', 
               'reptiles', 
               'small_mammals', 
               'trees', 
               'vehicles_1', 
               'vehicles_2']

def get_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(img_tensor):
    # Normalize to [0,1]
    img_tensor /= 255.  
    return img_tensor

def preprocess_image(img):
    img = prepare_image(img, target_size) # Resize the image
    img_tensor = np.array(img, dtype='float32')
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)
    return img_tensor

def predict(url):
    img = get_image(url)
    img_tensor = preprocess_image(img)
    interpreter = tflite.Interpreter(model_path='cifar-100-model.tflite')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, img_tensor)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    return dict(zip(class_names, float_predictions))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result