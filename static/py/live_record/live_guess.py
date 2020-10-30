#See Jupyter Notebook AudioRecord.ipynb for more detailed breakdown

#all imports first
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import seaborn as sns
import parselmouth as PM
import librosa
import librosa.display
import pandas as pd
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
from numpy import load

#function for recording an .wav audio file
def record():
    fs = 44100
    seconds = 5
    recording = sd.rec(int(seconds*fs), samplerate=fs, channels=2)
    sd.wait()
    write('../livefile.wav', fs, recording)

#function for playing back/listening to created audio file
def playback():
    file = "../livefile.wav"
    data, fs = sf.read(file, dtype = 'float32')
    sd.play(data, fs)
    status = sd.wait()

#function for creating chart of audio file, and putting created recording through model
def emo_guess():

    #creates chart
    data, sampling_rate = librosa.load("../livefile.wav")
    plt.figure(figsize=(15,5))
    librosa.display.waveplot(data, sr = sampling_rate)
    plt.savefig("live_librosa_chart.png")

    #creates data frame of features
    X, sample_rate = librosa.load('../livefile.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    live_mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    live_feature = live_mfccs
    live = live_feature
    live= pd.DataFrame(data=live)
    live = live.stack().to_frame().T

    #pushes dataframe features through model to get output
    json_file = open('../model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../saved_models/Emotion_Voice_Detection_Model.h5")
    twodim= np.expand_dims(live, axis=2)
    livepreds = loaded_model.predict(twodim, 
                            batch_size=32, 
                            verbose=1)
    livepreds1=livepreds.argmax(axis=1)
    liveabc = livepreds1.astype(int).flatten()
    print(liveabc)
    lb = LabelEncoder()
    y_train=load('../y_train.npy',allow_pickle=True)
    y_test=load('../y_test.npy',allow_pickle=True)
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    livepredictions = str(lb.inverse_transform((liveabc))[0])
    gender_emotion = livepredictions.split('_')
    gender=gender_emotion[0].capitalize()
    emotion=gender_emotion[1].capitalize()
    print(gender, emotion)
