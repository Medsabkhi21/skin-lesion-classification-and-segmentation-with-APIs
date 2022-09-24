import io
from numpy import dtype, float32
import tensorflow as tf
from features  import Features
import efficientnet.tfkeras as efn
import numpy as np

def one_hot_encode(feature):
     location = ['head/neck' ,'upper extremity', 'lower extremity', 'torso' , 'palms/soles','oral/genital']
     encoded_feature = [ 1 if k==feature else 0  for k in location]
     return encoded_feature

def prepare_features(features:Features):
    
    anatom_site_encoded = one_hot_encode(features['anatom_site'])
    one_hot_site = tf.reshape(tf.convert_to_tensor(anatom_site_encoded, dtype=tf.float32), [1,6])
    age_min = 0
    age_max = 100
    age = tf.reshape(tf.cast((features['age']-age_min)/(age_max-age_min), tf.float32), [1,1])
    sex = tf.reshape(tf.cast(features['sex'], tf.float32), [1,1])
    
    features = tf.keras.layers.Concatenate()([sex,age,one_hot_site])
    features = tf.reshape(features,[1,8])
    
    return features

def prepare_image(img, dim=384):    
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [dim,dim])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [1,dim,dim, 3])
    return img



def get_model():
    model = tf.keras.models.load_model(
        '/model/pb'
    )
    print(model.summary())
    return model


                     
def build_model(dim=384):

    input_image = tf.keras.layers.Input(shape=(dim,dim,3))
    input_features = tf.keras.layers.Input(shape=(8))
    
    base = efn.EfficientNetB5(input_shape=(dim,dim,3),include_top=False)
    
    x = base(input_image, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(units = 128, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Train the feature map with a shallow dense layer
    x2 = tf.keras.layers.Dense(units = 16, activation="relu")(input_features)
    x2 = tf.keras.layers.Dropout(0.3)(x2)

    # concatenate outputs of two branches
    x = tf.keras.layers.concatenate([x, x2])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(units = 128, activation="relu")(x) 
    x = tf.keras.layers.Dropout(0.5)(x)

    # make predictions
    x = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.Model(inputs = [input_image, input_features],outputs=x)
    model.load_weights("model/weights_lots-of-training-efb5.h5")
    model.summary()
    return model


def get_binary(img,features:Features):
    features = features.dict()
    preprocessed_features = prepare_features(features)
    preprocessed_img = prepare_image(img)
    #model = get_model()
    model = build_model()
    prediction = model([preprocessed_img,preprocessed_features])
    prediction = prediction.numpy().tolist()
    return prediction[0]
    
  


