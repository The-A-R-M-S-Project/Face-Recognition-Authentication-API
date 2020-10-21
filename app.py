from flask import Flask, jsonify,request
# from database.db import initialize_db
# from resources.admin import admin
from utils import generate_embeddings
from utils import detect_face
import pickle
import numpy as np
from flask_cors import CORS
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import base64

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return '<h1>Home</h1>'

@app.route('/predict', methods=["POST"])
def face_recognition():
    try:
        body = request.get_json()
        datauri = body['datauri']
        datauri = datauri.partition(',')[2]
        img = open('userface.png', 'wb')
        img.write(base64.b64decode(datauri))
        img.close()
        face = detect_face.detect_face('./userface.png')
        face_embeddings = generate_embeddings.generate_embeddings(face)
        # importing facenet-classifier
        classifier = pickle.load(open('models/classifier-facenet.sav', 'rb'))
        prediction = classifier.predict(face_embeddings)
        print(prediction)
        return {"prediction": str(prediction[0])}
    except:
        return {"message": "Something went wrong at the backend"}
app.run(debug=True)
