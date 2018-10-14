import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model_path = 'trained_model/weights_0.hdf5'

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


model = load_model(model_path)

img = image.load_img('cricket-test.jpg', target_size=(299,299))
preds = predict(model, img)

print (preds)


img = image.load_img('baseball-test.jpeg', target_size=(299,299))
preds = predict(model, img)

print (preds)
