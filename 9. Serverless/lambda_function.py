import tflite_runtime.interpreter as tflite
from PIL import Image
from io import BytesIO
from urllib import request
import numpy as np


#import model
interpreter = tflite.Interpreter(model_path='cats-dogs-v2.tflite')
interpreter.allocate_tensors()


# get input and output index
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def download_image(url):
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

# url = 'https://upload.wikimedia.org/wikipedia/commons/1/18/Vombatus_ursinus_-Maria_Island_National_Park.jpg'


def preprocessor(img):
    x = np.array(img, dtype='float32') / 255

    return np.array([x])


def predict(url):

    img = download_image(url)
    img = prepare_image(img, (150, 150))
    X = preprocessor(img)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()
    label = ['cat' if preds[0] < 0.5 else 'dog']

    return dict(zip(label, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
