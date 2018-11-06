from flask import Flask,request
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import base64 

app = Flask(__name__)

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds[0][0]


@app.route('/')
def index():
    return "Hello Sam"

@app.route('/imageclass', methods=['POST'])

def send_prediction():
    print "POST works"
    f = request.files['file']
    f.save('uploads/'+f.filename)
    img = image.load_img('uploads/'+f.filename, target_size=(299,299))

    model_path = 'trained_model/weights_0.hdf5'
    model = load_model(model_path)

    pred = predict(model, img)

    return str(pred)

if __name__ == "__main__":

    print("Starting server")
    app.run(host = "0.0.0.0", port = 5000, debug=True)