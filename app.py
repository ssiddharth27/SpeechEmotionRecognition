from flask import Flask,render_template
from flask import Flask, render_template, request, redirect
import soundfile
from keras.models import load_model
import librosa
import numpy as np
# from fastai import *                                 
# from fastai.vision.all import *
# from fastai.vision.data import ImageDataLoaders
# from fastai.tabular.all import *
# from fastai.text.all import *
# from fastai.vision.widgets import *
# import torch
# torch.cuda.empty_cache()
import pickle

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

model = load_model('best.hdf5')
#m1 = load_learner("speech_model_1.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    c = ""
    l = ['Neutral','Calm','Happy','Sad','Angry','Fearful','Disgust','Surprised']
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        # if file:
        #     y, sr = librosa.load(file)
        #     yt,_=librosa.effects.trim(y)
        #     audio_spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=100)
        #     audio_spectogram = librosa.power_to_db(audio_spectogram, ref=np.max)
        #     librosa.display.specshow(audio_spectogram, y_axis='mel', fmax=20000, x_axis='time')

        #     p = os.path.join("D:\deploy\live_image", "{}.jpg".format(1))
        #     plt.savefig(p)
        #     is_angry,a, probs = m1.predict(p)
        #     c = c+is_angry
            

        if file:
            with soundfile.SoundFile(file) as audio:
                waveform = audio.read(dtype="float32")
                sample_rate = audio.samplerate
                stft_spectrogram=np.abs(librosa.stft(waveform))
                chromagram=np.mean(librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate).T,axis=0)
                melspectrogram=np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000).T,axis=0)
                mfc_coefficients=np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40).T, axis=0) 

                feature_matrix = np.hstack((chromagram, melspectrogram, mfc_coefficients))

            b = [feature_matrix]
            b = np.array(b)
            a = model.predict(b)
           
            xt = np.argmax(a)
            c = c + l[xt]

    return render_template('index2.html', output=c)



if __name__ == "__main__":
    app.run(debug=True, threaded=True)