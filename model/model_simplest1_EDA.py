import numpy as np
import pandas as pd
import librosa
# import lasagne
# from lasagne import layers
# from lasagne.updates import nesterov_momentum
# from nolearn.lasagne import NeuralNet
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
# i had to do pip install tensorflow
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler


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
    return sig_splits[:-1]
        # even tho the int function rounds DOWN, it will still create a split at the end such that the last split is <3 sec, so I return everything but the last split
        # ex: incoming data is 10 sec sampled at 100. So there are 1000 numpy values. It will be the same as doing range(3) bc it will go 0 - 1000 with stepsize of 3*100 = 300 stepsize.



if __name__ == '__main__':
    sig, sr = librosa.load('../example/Soundscape_1.wav', sr=1000) # a 10 sec sample
    sig_splits = split_signal(sig, rate=1000)
    sig_splits = np.asarray(sig_splits) # i could just have split_signals spit out the whole thing as an array..
    labels_name = ['norcar', 'rewbla', 'rewbla', 'norcar', 'norcar', 'rewbla', 'norcar', 'orcori', 'norcar', 'norcar', 'dowwoo', 'dowwoo', 'euptit1', 'norcar', 'rewbla', 'dowwoo', 'rewbla', 'rewbla', 'norcar', 'norcar']
    labels_num = [1,2,2,1,1,2,1,3,1,1,4,4,5,1,2,4,2,2,1,1]
    labels_num = np.asarray(labels_num)
 # 5 unique- 5 categories
 # even tho this file is already in 3 sec clips, I'm going to run it through the sig_splits to make sure it is EXACTLY 3 sec each so that the NN doesn't get messed up.
    print(len(sig_splits))  #should be 20- yes it is
    # gonna add a quick filter here which drops any 3 sec audio which has an avg Hz < 400 and I'll have to make sure I also drop corresponding labels
# building a model: https://lasagne.readthedocs.io/en/latest/user/tutorial.html
# https://martin-thoma.com/lasagne-for-python-newbies/
# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
# https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/


    #create model - can't do CNN until i have a 2d dataset instead of 1d
    # model = Sequential()
    # #add model layers
    # model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(3000,1)))
    # model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(10, activation='softmax'))

    m_inputs = keras.Input(shape=(3000,))
    x = layers.Dense(3000, activation='relu')(m_inputs)
    x = layers.Dense(25, activation='relu')(x)
    m_outputs = layers.Dense(1, activation='softmax',name='predictions')(x)


    model = keras.Model(inputs=m_inputs, outputs=m_outputs)
    # model = Sequential(inputs=m_inputs, outputs=m_outputs)
    # model.add(Dense(3000, input_dim=1, activation='relu')) # the first one has to be the same dimensions as incoming data
    # model.add(Dense(4, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(sig_splits, labels_num.reshape(20,1), epochs=2, verbose=0)
    print(model.predict(sig_splits))
