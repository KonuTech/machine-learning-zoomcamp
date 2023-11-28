#!/usr/bin/env python
# coding: utf-8

from io import BytesIO
from urllib import request

import numpy as np
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
from PIL import Image

MODEL_NAME = "bees-wasps-v2"

interpreter = tflite.Interpreter(model_path=f"{MODEL_NAME}.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img


def prepare_input(x):
    return x * (1.0 / 255)


def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))

    X = np.array(img, dtype="float32")
    X = np.array([X])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    return float(preds[0, 0])


def lambda_handler(event, context):
    url = event["url"]
    pred = predict(url)
    result = {
        "prediction": pred
    }

    return result
