# PRML-CSL2050 Course Project

## Problem Statement
-------------------------------------------------
Speech Emotion Recognition, abbreviated as SER, is the act of attempting to recognize human emotion and affective states from speech. This is capitalizing on the fact that voice often reflects underlying emotion through tone and pitch. In this project We particularly focused on feature engineering techniques for audio data and provide an in-depth look at the logic,
concepts, and properties of the Multilayer Perceptron (MLP) model, an ancestor and the
origin of deep neural networks (DNNs) today. We also provide an introduction to a few basic
machine learning models.

## Data
-------------------------------------------------
Data can be found at the given [link](https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view).

Data consist of audio files. Audio is sourced from 24 actors (12 male, 12 female) repeating two sentences with a variety of emotions and intensity. In total data consist of 1440 speech files (24 actors * 60 recordings per actor).

## Code Structure
-------------------------------------------------
1.   Intro: Speech Emotion Recognition on the RAVDESS dataset

2.   Feature Extraction
     
     Load the Dataset and Compute Features
     
     Feature Scaling

3.   Feature Engineering
    
     Mel-Frequency Cepstral Coefficients
    
     Mel Spectrograms and Mel-Frequency Cepstrums
     
     The Chromagram

4.   Classical Machine Learning Models with Accuracy( Precision, Recall, F-Score )

      k Nearest Neighbours
    
      Random Forests Classifier
    
      XGB Classifier

5.   Feature extraction 

     Graphical spectrogram for all data
     
     Visulize image representations of the audio

6.   Training and Evaluating the Deep Neural network(DNN) Model 
    
     Neural Network Model
    
     Resnet model with live audio speech Recognition

     Evalution (The Confusion Matrix, Precision, Recall, F-Score)
     
## How to run code!
-------------------------------------------------
• Code for model is avalable at the given [link](https://colab.research.google.com/drive/1NRSI7CJATXf_pRxT_meILqH5XrB84n_k?authuser=1).

• Nothing is needed to be pre-installed to run the code.

• Go to google colab through the above link.

• Firstly, you need to mount data stored in your drive to google colab.

• And you need to modify path of data according to the location of it in your drive.

• You will have to manually make the directories with name output_folder_train and output_folder_test containing sub-directories as emotions name (neutral, calm, happy, sad, angry, fearful, disgust, surprised).

• Now you can smoothly run the rest of the cells. Code working is instructed there in comment statements.

• For live speech emotion recognition part you need to terminate the recording, which will start just after running the get audio cell.

## Modules Required
-------------------------------------------------
• Scikit-learn: ML library used

• Tensorflow Keras: ML library used

• Librosa: Python package for music and audio analysis.

• Scipy.io.wavfile: Return the sample rate (in samples/sec) and data from a WAV file.

• Glob: Used to return all file paths that match a specific pattern.

• Fastai: Fastai is a deep learning library that provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains. We will be using
fastai.vision.

• Pandas: Used to make DataFrame.

• Matplotlib: This allows us to plot spectrograms

• Tkinter: Tkinter is the standard GUI library for Python. Python when combined with Tkinter provides
a fast and easy way to create GUI applications.

## Team members
-------------------------------------------------
Aditi Tiwari (B20EE005)

Shreya Sachan (B20EE065)

Siddharth Singh (B20EE067)

