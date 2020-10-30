import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import librosa
import glob
import librosa.display
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
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write


#function for recording an .wav audio file, records 5 seconds
def record():
    fs = 44100
    seconds = 5
    recording = sd.rec(int(seconds*fs), samplerate=fs, channels=2)
    sd.wait()
    write('livefile.wav', fs, recording)

#function for playing back/listening to created audio file
def playback():
    file = "livefile.wav"
    data, fs = sf.read(file, dtype = 'float32')
    sd.play(data, fs)
    status = sd.wait()

def main():
    file = "livefile.wav"

    sns.set() # Use seaborn's default style to make attractive graphs

# Plot nice figures using Python's "standard" matplotlib library
    snd = parselmouth.Sound(file)
    plt.figure(figsize=(15, 5))
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    #plt.show() or plt.savefig("Resources/images/sound.png")
    plt.savefig("static/images/sound.png")



    def draw_spectrogram(spectrogram, dynamic_range=70):
        X, Y = spectrogram.x_grid(), spectrogram.y_grid()
        sg_db = 10 * np.log10(spectrogram.values)
        plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
        plt.ylim([spectrogram.ymin, spectrogram.ymax])
        plt.xlabel("time [s]")
        plt.ylabel("frequency [Hz]")

    def draw_intensity(intensity):
        plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
        plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
        plt.grid(False)
        plt.ylim(0)
        plt.ylabel("intensity [dB]")

    intensity = snd.to_intensity()
    spectrogram = snd.to_spectrogram()
    plt.figure(figsize=(15, 5))
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_intensity(intensity)
    plt.xlim([snd.xmin, snd.xmax])
    plt.savefig("static/images/spectrogram.png")



    def draw_pitch(pitch):
        # Extract selected pitch contour, and
        # replace unvoiced samples by NaN to not plot
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values==0] = np.nan
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
        plt.grid(False)
        plt.ylim(0, pitch.ceiling)
        plt.ylabel("fundamental frequency [Hz]")

    pitch = snd.to_pitch()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pre_emphasized_snd = snd.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
    plt.figure(figsize=(15, 5))
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_pitch(pitch)
    plt.xlim([snd.xmin, snd.xmax])
    plt.savefig("static/images/spectrogram_0.03.png")


    #livedf= pd.DataFrame(columns=['feature'])
    X, sample_rate = librosa.load(file, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive

    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    livedf2

    
    json_file = open('static/py/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("static/py/saved_models/Emotion_Voice_Detection_Model.h5")

    twodim= np.expand_dims(livedf2, axis=2)

    livepreds = loaded_model.predict(twodim, 
                            batch_size=32, 
                            verbose=1)

    livepreds1=livepreds.argmax(axis=1)

    liveabc = livepreds1.astype(int).flatten()
    print(liveabc)
    lb = LabelEncoder()
    y_train=load('static/py/y_train.npy',allow_pickle=True)
    y_test=load('static/py/y_test.npy',allow_pickle=True)
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    livepredictions = str(lb.inverse_transform((liveabc))[0])
    gender_emotion = livepredictions.split('_')
    gender=gender_emotion[0].capitalize()
    emotion=gender_emotion[1].capitalize()
    # result={
    #     'gender':gender,
    #     'emotion':emotion
    # }
    return gender,emotion

emotion= main()
# waveform.show()
# print(gender)
print(emotion)