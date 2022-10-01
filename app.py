import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import PIL


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_features(img_path,model):
    img = tensorflow.keras.preprocessing.image.load_img(img_path,target_size=(224,224))
    #Img to array
    img_array = image.img_to_array(img)
    #keras need data in  batches, so converting single image batch
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # Resnet preprcess the image before making any predictions
    preprocessed_img = preprocess_input(expanded_img_array)
    # Predicting the by using model
    result = model.predict(preprocessed_img).flatten()
    # Noemalizing
    normalized_result = result / norm(result)

    return normalized_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))

