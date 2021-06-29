import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt


#
#
# y, sr = librosa.load(librosa.ex('trumpet'))
# f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
# times = librosa.times_like(f0)
#
# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# fig, ax = plt.subplots()
# img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
# ax.set(title='pYIN fundamental frequency estimation')
# fig.colorbar(img, ax=ax, format="%+2.f dB")
# ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
# ax.legend(loc='upper right')
# # plt.show() # works- cool!
#
# no_na_f0 = f0[~np.isnan(f0)]
# print(no_na_f0)
# print('*'*20)
# print(no_na_f0.mean())
# print('*'*20)

# print(type(f0))
# print(f0)



# i know the pitch of my speaking voice is about 100 -200 Hz.
#https://www.onlinemictest.com/tuners/pitch-detector/
# so I want to record my voice and see if it correctly gives an avg which makes sense.
# ----------------------------------
 # i just recorded something quickly in .wav here: https://voicecoach.ai/voice-recorder
y, sr = librosa.load('../example/meg2.wav')
# y, sr = librosa.load('../example/puaiohi_sample.wav', sr=10000) # really small, short, high chirps around #1800 Hz- avg is 1812.5Hz - even that alone will be able to identify it compared to other birds in the area. There's probably not many birds which do small chirps. there were a couple of repeated chirps right around 1920 - 1940 Hz. Maybe Puaiohi is the ONLY bird in the area which does repeated, small chirps around 1920 - 1940 Hz? the NN will be able to easily find rhythms too. But Hz is great. To our ear, its hard to tell small differences. But an NN will be able to dial in on the exact pitches Puaiohi tend to do that are unique to that bird. NNs are great at memorizing. Since we have hours of recordings, it will see which ranges and pitches and ryhthms are unique to Puaiohi. Even tho each bird has a broad range, they tend to like to hit certain notes. Every bird is different, so the more birds, the better. But some calls are the same and quite identical in pitch (think- mourning dove). It should also easily see if the pitch tends to bend up or down for certain calls.
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=30, fmax=10000)
times = librosa.times_like(f0)
print(len(f0))
print(len(times))
# f0 is the fundamental Hz- times gives me the times in seconds corresponding to the pitch. So- i could create a numpy array of all 0s for each 3 sec clip, and then add the pitch from start to finish for each time chunk and append that to the raw numpy, and give both to the NN.

# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# fig, ax = plt.subplots()
# img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
# ax.set(title='pYIN fundamental frequency estimation')
# fig.colorbar(img, ax=ax, format="%+2.f dB")
# ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
# ax.legend(loc='upper right')
# print(len(f0))
# print(len(y))
# no_na_f0 = f0[~np.isnan(f0)]
# # print(no_na_f0)
# print(len(no_na_f0))
# print('*'*20)
# print(no_na_f0.mean())
# print('*'*20)
# plt.show() # works- cool! oh wow thats so great! I had a lot of space in my voice and it didnt avg the space- maybe they were nans that i deleted? anyway, its great that it only took the avg of the actual sound, not some bullshit microphone or background noise- just the main sound of my voice. it says the avg is 119.76Hz which sounds exactly true for my voice, and the spectrogram makes sense too- yay!
