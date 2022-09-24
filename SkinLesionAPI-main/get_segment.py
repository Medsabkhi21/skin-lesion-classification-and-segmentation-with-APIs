import tensorflow as tf
import numpy as np
from PIL import Image 
import io
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy



def dice_bce_loss(y_true, y_pred):
    bce_loss = binary_crossentropy(y_true, y_pred)
    dice_loss = soft_dice_loss(y_true, y_pred)
    
    return bce_loss + dice_loss

def dice_coefficient(y_true, y_pred, epsilon = 0.000001):
    dice_numerator = K.sum(x = y_true*y_pred, axis = (1,2))
    dice_denominator = K.sum(x = y_true, axis = (1,2)) + K.sum(x = y_pred, axis = (1, 2)) + epsilon
    dice_coefficient = K.mean((2*dice_numerator + epsilon) / dice_denominator, axis = 0)

    return dice_coefficient

def get_segmentor():
    model = tf.keras.models.load_model(
        'model/unet_checkpoint.hdf5', 
        custom_objects={'dice_bce_loss' : dice_bce_loss, 'dice_coefficient' : dice_coefficient}
    )
    return model

def enhance(img):
    sub = img.flatten()
    for i in range(len(sub)):
        if sub[i] > 0.5:
            sub[i] = 1
        else:
            sub[i] = 0
    return sub

def get_segments(model, binary_image):
   
    input_image = Image.open(io.BytesIO(binary_image)).convert('RGB')
    ori_size = input_image.size
    input_image = input_image.resize((256, 256))
    input_image = np.array(input_image)
    input_image = input_image / 255.0
    input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])

    with tf.device('/CPU:0'):
        segmented_image = model(input_image, training=False)

    seg_img_shape = (segmented_image.shape[1], segmented_image.shape[2])
    segmented_image = segmented_image.numpy()
    segmented_image = enhance(segmented_image)
    segmented_image = segmented_image.reshape(seg_img_shape[0],seg_img_shape[1])
    
    segmented_image = segmented_image * 255.0
    segmented_image = segmented_image.astype(np.uint8)

    segmented_image = Image.fromarray(segmented_image)
    segmented_image = segmented_image.resize(ori_size)
    
    return segmented_image