from flask import Flask, jsonify
import numpy as np
import json
import requests
from flask import render_template
from model_train import record, playback, main

app = Flask(__name__)

@app.route('/record')
def record1():

#function for recording an .wav audio file, records 5 seconds
    record()
    return render_template("predict_initial.html")

@app.route('/playback')
def playback1():
    playback()
#function for playing back/listening to created audio file
    return render_template("predict_initial.html")

@app.route('/predict')
def model():

    gender,emotion=main()
    print(emotion)
    print(gender)
# Voice regognition and emotion prediction
    return render_template("predict.html",emotion=emotion,gender=gender)

# Define what to do when a user a specific route
@app.route("/")
def index1():
    return render_template("predict_initial.html")

@app.route("/data.html")
def data():
    return render_template("data.html")

@app.route("/about.html")
def about():
    # print("Server received request for 'Home' page...")
     return render_template("about.html")

@app.route("/code_walk_through.html")
def license():
    return render_template("code_walk_through.html")

@app.route("/predict.html")
def index2():  
    return render_template("predict.html")
    
@app.route("/predict_initial.html")
def index3():  
    return render_template("predict_initial.html")

# run app
if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000,threaded=False)
