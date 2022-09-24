import tensorflow as tf
import numpy as np
from PIL import Image 
import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D



def create_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size = (3,3), input_shape = (28, 28, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

    model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))
    model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    model.load_weights("model/classifier.hdf5")
    model.summary()
    return model;


def preprocess(binary_image):
    input_image = Image.open(io.BytesIO(binary_image)).convert('RGB')
    input_image = input_image.resize((28, 28))
    input_image = np.array(input_image)
    input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    return input_image

def dict_map(prediction):
    lesion_type_dict =  [ 'nv',
     'mel',
     'bkl',
     'bcc',
     'akiec',
     'vasc',
     'df']
     
    return dict(zip(lesion_type_dict,prediction))


def get_class(binary_image):
    model = create_model()
    input_image = preprocess(binary_image)
    with tf.device('/CPU:0'):
        prediction = model.predict(input_image)
        
    prediction = prediction.tolist()
    result = dict_map(prediction[0])

    return result
    
    