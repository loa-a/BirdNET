import numpy as np
import pandas as pd
import librosa
import os
import glob
import itertools
# import lasagne
# from lasagne import layers
# from lasagne.updates import nesterov_momentum
# from nolearn.lasagne import NeuralNet
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import np_utils
# i had to do pip install tensorflow
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
# installing pydub to convert mp3 to wav

from os import path
import pydub
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from datetime import datetime

np.random.seed(42)

RATE = 10000 #making this global since it affects many things and we may want to be able to change it easily. during testing its nice to lower it temporarily

# files
# src = "transcript.mp3"
# dst = "test.wav"
#
# # convert wav to mp3
# sound = AudioSegment.from_mp3(src)
# sound.export(dst, format="wav")
#
# combined = AudioSegment.empty()
# for song in playlist_songs:
#     combined += song
#
# combined.export("/path/output.wav", format="wav")


# I'm taking my proof of concept simple 'model_simplest1_EDA.py' and even tho its shitty- training it on 3 birds from the kaggle competition and seeing if it finds anything.
# I like to start out shitty and basic, because then, for every improvement, I can see how much the improvement helps
# I'm using bluejay "blujay", baltimore oriole "balori" and yellow warbler "yerwar"

# ----------------  load data --------------------------------------
# I'm having trouble with using AudioSegment and ffmpeg- need to move the file etc- for now i'm just batch converting each mp3 folder to wav using Audacity
#https://forum.audacityteam.org/viewtopic.php?t=63821
def load_and_convert(input_path, folder, input_type='wav', output_type='wav', output_path='../example/'):
    '''
    INPUT: input_path = path needed to get to the folder
    INPUT: folder = folder name (as a string) that has lots of bird sounds of one species
    INPUT: input_type = file type to grab ('mp3' or 'wav')
    INPUT: output_type = requested filetype of output
    INPUT: output_path = requested output folder- include '/'s
    OUTPUT: folder_combined.output_type; combined file with requested file type

    Since it looks like most of our data will be one bird species at a time in a folder of multiple files, this will go in the requested folder, grab all the files, concat them, and export them as one large audio file which we can split later. Librosa wants everything in wav so I make sure to convert it to wav too.
    '''
    print('starting load ')
    file_list = glob.glob(os.path.join(os.getcwd(), input_path+folder, '*.'+input_type))
    # print(file_list)
    playlist = []
    combined = AudioSegment.empty()

    if input_type == 'wav':
        for f in file_list:
            playlist.append(AudioSegment.from_wav(f))

    if input_type == 'mp3':
        for f in file_list:
            playlist.append(AudioSegment.from_mp3(f))

    for song in playlist:
        combined += song

    combined.export(output_path+folder+'_combined.wav', format="wav")
    print('file saved')





def split_signal(sig, rate=48000, seconds=3):
    '''
    INPUT: sig = processed numpy list from a a large (>3 sec) wav- its just raw compressions the microphone felt, sampled at the sampling rate
    INPUT: rate = sampling rate. If a 1 sec audio wav was sampled at a rate of 48,000, then there will be 48,000 numpy values per second of audio, so we need to know the sampling rate that was used to create the incoming sig
    INPUT: seconds = desired, set # of seconds to give to the NN- the NN requires each be the same exact length. BirdNET used 3 s and it works well for them. I think that is pretty reasonable. If we say like 10, well then that severely limits future incoming audio, which may be shorter. If we say 0.1 sec or something, that won't be enough for the NN to learn things involving rhythm.
    OUTPUT: list of numpy signals, split by the seconds input, with everything exactly the same length- the last audio clip <3 sec is not returned
    '''
# Split signal
    sig_splits = []
    for i in range(0, len(sig), int(seconds*rate)):
        split = sig[i:i + int(seconds * rate)]
        sig_splits.append(split)
    return np.asarray(sig_splits[:-1])
        # even tho the int function rounds DOWN, it will still create a split at the end such that the last split is <3 sec, so I return everything but the last split
        # ex: incoming data is 10 sec sampled at 100. So there are 1000 numpy values. It will be the same as doing range(3) bc it will go 0 - 1000 with stepsize of 3*100 = 300 stepsize.

def choose_best(splits, cutoff=800):
    '''
    INPUT: splits = signal numpy splits
    INPUT: cutoff = avg pitch in Hz that the clip must be higher than
    OUTPUT: best signal numpy splits
    It only returns splits with a pitch higher than the cutoff. This is to help train the NN on quality data, and also save time during NN fitting by cutting down on excessive data. We want the training and production data to be quality, and ignore clips that are just silent or mostly wind etc. Birds sing around 1000 - 8000 Hz, much higher than humans or most other noises on Earth luckily, so we are taking advantage of that
    Warning- this takes a long time. An hr of audio will take a couple hrs to process...
    '''
    best_splits = []
    for split in splits:
        f0, voiced_flag, voiced_probs = librosa.pyin(split, fmin=30, fmax=10000)
        no_na_f0 = f0[~np.isnan(f0)]
        if no_na_f0.mean() > cutoff:
            best_splits.append(split)

    return np.asarray(best_splits)

def build_model_object(shape):
    '''
    work in progress- ill make it more customizable later and also make a CNN version
    shape is the incoming dataset's shape
    '''
    model = Sequential()
    model.add(Dense(shape, input_dim=shape, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




if __name__ == '__main__':
    t1 = datetime.now()
    # load the files
    # load_and_convert('../example/', 'balori_wav') # ill make this label 1
    # load_and_convert('../example/', 'blujay_wav') # 2
    # load_and_convert('../example/', 'yerwar_wav')  # 3

    # vectorize and split the files
    # balori_sig, sr = librosa.load('../example/balori_wav/XC11476.wav', sr=RATE)
    balori_sig, sr = librosa.load('../example/balori_wav_combined.wav', sr=RATE)
    balori_splits = split_signal(balori_sig, rate=RATE)
    print(balori_splits)
    print(len(balori_splits))
    # blujay_sig, sr = librosa.load('../example/blujay_wav/XC16897.wav', sr=RATE)
    blujay_sig, sr = librosa.load('../example/blujay_wav_combined.wav', sr=RATE)
    blujay_splits = split_signal(blujay_sig, rate=RATE)
    print(blujay_splits)
    print(len(blujay_splits))
    # yerwar_sig, sr = librosa.load('../example/yerwar_wav/XC17028.wav', sr=RATE)
    yerwar_sig, sr = librosa.load('../example/yerwar_wav_combined.wav', sr=RATE)
    yerwar_splits = split_signal(balori_sig, rate=RATE)
    print(yerwar_splits)
    print(len(yerwar_splits))
     #took 8 min @ 48k RATE

# --------- optional - take out worst audio clips ---------------
    # print(datetime.now())
    # #choose the best audioclips and discard the rest
    # balori_splits = choose_best(balori_splits)
    # print(len(balori_splits)) # these incoming files are pretty good so there may not be many taken out.
    # blujay_splits = choose_best(blujay_splits)
    # print(len(blujay_splits))
    # yerwar_splits = choose_best(yerwar_splits)
    # print(len(yerwar_splits))
    print(datetime.now())

    # create train-test split - keeping everything as a numpy array
    y_balori = [1]*(len(balori_splits))
    # print(y_balori)
    y_blujay = [2]*(len(blujay_splits))
    # print(y_blujay)
    y_yerwar = [3]*(len(yerwar_splits))
    # print(y_yerwar)
    y_orig = list(itertools.chain(y_balori, y_blujay, y_yerwar))
    # print(y_orig)
    y_orig = np.asarray(y_orig)
    dummy_y = np_utils.to_categorical(y_orig) #making dummies
    # print(y_orig)
    # print(dummy_y)
    X_orig = np.vstack((balori_splits, blujay_splits, yerwar_splits))
    # print(X_orig)
    # this sklearn function shuffles the order, which is good for NN training
    X_train, X_test, y_train, y_test = train_test_split(X_orig, dummy_y, test_size=0.33, random_state=42) #using only 10% to fit for now since it takes so fucking long

    # fit, predict, and check val
    # https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
    model = build_model_object(RATE*3) #since rate*3 is how long each numpy row is since we are sticking with exactly 3 sec of audio
    model.fit(X_train, y_train, epochs=3, verbose=1) #.reshape(20,1)
    pred = model.predict(X_test) # lots of parameters to play around with- more epochs helps but takes longer. We should probably give up on this and do CNN- i was just curious how it would do- maybe a NN model based solely on pitch changes would be good.
    print(pred)
    print(y_test)
    pred_df = pd.DataFrame(pred)
    actual_df = pd.DataFrame(y_test)
    answer = pd.concat([pred_df,actual_df], axis=1)
    print(pred_df)
    print('sum score of balori predicted: ', pred_df.iloc[:,1].sum())
    print('precision: ', average_precision_score(actual_df.iloc[:,1],pred_df.iloc[:,1]))
    print('sum score of blujay predicted: ', pred_df.iloc[:,2].sum())
    print('precision: ', average_precision_score(actual_df.iloc[:,2],pred_df.iloc[:,2]))
    print('sum score of yerwar predicted: ', pred_df.iloc[:,3].sum())
    print('precision: ', average_precision_score(actual_df.iloc[:,3],pred_df.iloc[:,3]))
    # print('# unique predictions: ', len(np.unique(pred)))
    # print('# of 1s predicted, balori: ', len(np.where(pred==1)[0]))
    # print('# of 2s predicted, blujay: ', len(np.where(pred==2)[0]))
    # print('# of 3s predicted, yerwar: ', len(np.where(pred==3)[0]))
    # good to check - sometimes w NN, it just predicts the same value for everything
    # print('overall accuracy: ', accuracy_score(y_test, pred))
    # print('overal precision score: ', average_precision_score(y_test,pred))
    # for 3 birds w roughly equal audiofiles, a purely avg guess would be 33% accurate, so it better beat 0.33!
    # it would be a good idea to check the accuracy per bird - also play around with a probability score instead of just the categorical score. sometimes if we set a cutoff of like >80% in the probability, then the accuracy is really good for that chunk- thats also maybe how we will use it in the use-case- give users the bird with the highest probability and the actual probability score.

    answer.to_csv('model_scores.csv')
    print(answer)
    t2 = datetime.now()
    print('*'*20)
    print('this crazy mutha took this long: ', t2-t1)
    print('and you used this RATE: ', RATE)
