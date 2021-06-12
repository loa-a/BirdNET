# the goal of this sloppy program is to reproduce what BirdNET does: to take some labelled bird audio files, and be able to build a model and make predictions. I'm using a small dataset, so the accuracy will be poor, but I just want to see if I can get it to work - Lance Culnane


# note- the code from BirdNET will fail bc theano got rid of 'downsample'. so one can either use an old version of theano (0.7) or the newest, in-development version of lasagne.

# I'm choosing to use the newest, yet to be released version of lasagne to fix it:
# pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
# see : https://github.com/aigamedev/scikit-neuralnetwork/issues/235

# For now, I like combining all the code into one so I can look at the functions

import sys
sys.path.append('..')

import os
import pickle
import operator
import numpy as np
import pandas as pd

import theano
import theano.tensor as T

from lasagne import layers as l
from lasagne import nonlinearities as nl

import config as cfg
from utils import log


from builtins import range

import librosa

import scipy

import json, gzip
import datetime

######## from BirdNET config.py #########
# this is the part I don't understand- i think this makes the config variables, which are then referenced in the model etc

# BirdNET uses eBird checklist frequency data to determine plausible species
# occurrences for a specific location (lat, lon) and one week. An EBIRD_THRESHOLD
# of 0.02 means that a species must occur on at least 2% of all checklists
# for a location to be considered plausible.
EBIRD_SPECIES_CODES = '../metadata/eBird_taxonomy_codes_2018.json'
EBIRD_MDATA = '../metadata/eBird_grid_data_weekly.gz'
USE_EBIRD_CHECKLIST = True
EBIRD_THRESHOLD = 0.02
DEPLOYMENT_LOCATION = (-1, -1)
DEPLOYMENT_WEEK = -1
GRID_STEP_SIZE = 0.25

# We use 3-second spectrograms to identify avian vocalizations.
# You can specify the overlap of consecutive spectrograms and the minimum
# length of a valid signal chunk (in seconds). You can also combine a number
# of extracted spectrograms for each prediction.
SPEC_OVERLAP = 0
SPEC_MINLEN = 1.0
SPECS_PER_PREDICTION = 1

# Adjusting the sigmoid sensitivity of the output layer can increase the
# number of detections (but will most likely also increase the number of
# false positives). You can set a minimum confidence threshold to suppress
# predictions with low score.

# The adjustment of the sigmoid sensitivity of the output layer can lead to an increase
# of detections (but will most likely also increase the number of false positives).
# You can set a minimum confidence threshold to suppress low score predictions.
SENSITIVITY = 1.0
MIN_CONFIDENCE = 0.1

# Loading a snapshot automatically sets the corresponding settings. Do not
# change these settings at runtime!
def setModelSettings(s):

    if 'classes' in s:
        global CLASSES
        CLASSES = s['classes']

    if 'spec_type' in s:
        global SPEC_TYPE
        SPEC_TYPE = s['spec_type']

    if 'magnitude_scale' in s:
        global MAGNITUDE_SCALE
        MAGNITUDE_SCALE = s['magnitude_scale']

    if 'sample_rate' in s:
        global SAMPLE_RATE
        SAMPLE_RATE = s['sample_rate']

    if 'win_len' in s:
        global WIN_LEN
        WIN_LEN = s['win_len']

    if 'spec_length' in s:
        global SPEC_LENGTH
        SPEC_LENGTH = s['spec_length']

    if 'spec_fmin' in s:
        global SPEC_FMIN
        SPEC_FMIN = s['spec_fmin']

    if 'spec_fmax' in s:
        global SPEC_FMAX
        SPEC_FMAX = s['spec_fmax']

    if 'im_dim' in s:
        global IM_DIM
        IM_DIM = s['im_dim']

    if 'im_size' in s:
        global IM_SIZE
        IM_SIZE = s['im_size']


######## end config.py ##################






######## from BirdNET grid.py #########################

##################### GLOBAL VARS #######################
GRID = []
CODES = []
STEP = 0.25

###################### LOAD DATA ########################
def load():

    global GRID
    global CODES
    global STEP

    if len(GRID) == 0:

        # Status
        log.p('LOADING eBIRD GRID DATA...', new_line=False)

        # Load pickled or zipped grid data
        if cfg.EBIRD_MDATA.rsplit('.', 1)[-1] == 'gz':
            with gzip.open(cfg.EBIRD_MDATA, 'rt') as pfile:
                GRID = json.load(pfile)
        else:
            with open(cfg.EBIRD_MDATA, 'rb') as pfile:
                GRID = pickle.load(pfile)

        # Load species codes
        with open(cfg.EBIRD_SPECIES_CODES, 'r') as jfile:
            CODES = json.load(jfile)

        STEP = cfg.GRID_STEP_SIZE
        print('DONE!')
        log.p(('DONE!', len(GRID), 'GRID CELLS'))

#################### PROBABILITIES ######################
def getCellData(lat, lon):

    # Find nearest cell
    for cell in GRID:
        if lat > cell['lat'] - STEP and lat < cell['lat'] + STEP and lon > cell['lon'] - STEP and lon < cell['lon'] + STEP:
            return cell

    # No cell
    return None

def getWeek():

    w = datetime.datetime.now().isocalendar()[1]

    return min(48, max(1, int(48.0 * w / 52.0)))

def getWeekFromDate(y, m, d):

    w = datetime.date(int(y), int(m), int(d)).isocalendar()[1]

    return min(48, max(1, int(48.0 * w / 52.0)))

def getSpeciesProbabilities(lat=-1, lon=-1, week=-1):

    # Dummy array
    p = np.zeros((len(cfg.CLASSES)), dtype='float32')

    # No coordinates?
    if lat == -1 or lon == -1:
        return p + 1.0
    else:

        # Get checklist data for nearest cell
        cdata = getCellData(lat, lon)

        # No cell data?
        if cdata == None:
            return p + 1.0
        else:

            # Get probabilities from checklist frequencies
            for entry in cdata['data']:
                for species in entry:

                    try:
                        # Get class index from species code
                        for i in range(len(cfg.CLASSES)):
                            if cfg.CLASSES[i].split('_')[0] == CODES[species].split('_')[0]:

                                # Do we want a specific week?
                                if week >= 1 and week <= 48:
                                    p[i] = entry[species][week - 1] / 100.0

                                # If not, simply return the max frequency
                                else:
                                    p[i] = max(entry[species]) / 100.0

                                break

                    except:
                        pass


        return p

def getSpeciesLists(lat=-1, lon=-1, week=-1, threshold=0.02):

    # Get species probabilities from for date and location
    p = getSpeciesProbabilities(lat, lon, week)

    # Parse probabilities and create white list and black list
    white_list, black_list = [], []
    for i in range(p.shape[0]):

        if p[i] >= threshold:
            white_list.append(cfg.CLASSES[i])
        else:
            black_list.append(cfg.CLASSES[i])

    return white_list, black_list
######## end grid.py ######################

######## from BirdNET audio.py for audio processing ###########
RANDOM = np.random.RandomState(1337)
CACHE = {}


def openAudioFile(path, sample_rate=48000, offset=0.0, duration=None):

    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, offset=offset, duration=duration, mono=True, res_type='kaiser_fast')

    return sig, rate

def noise(sig, shape, amount=None):

    # Random noise intensity
    if amount == None:
        amount = RANDOM.uniform(0.1, 0.9)

    # Create Gaussian noise
    noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)

    return noise

def buildBandpassFilter(rate, fmin, fmax, order=4):

    global CACHE

    fname = 'bandpass_' + str(rate) + '_' + str(fmin) + '_' + str(fmax)
    if not fname in CACHE:
        wn = np.array([fmin, fmax]) / (rate / 2.0)
        filter_sos = scipy.signal.butter(order, wn, btype='bandpass', output='sos')

        # Save to cache
        CACHE[fname] = filter_sos

    return CACHE[fname]

def applyBandpassFilter(sig, rate, fmin, fmax):

    # Build filter or load from cache
    filter_sos = buildBandpassFilter(rate, fmin, fmax)

    return scipy.signal.sosfiltfilt(filter_sos, sig)

def pcen(spec, rate, hop_length, gain=0.8, bias=10, power=0.25, t=0.060, eps=1e-6):
    s = 1 - np.exp(- float(hop_length) / (t * rate))
    M = scipy.signal.lfilter([s], [1, s - 1], spec)
    smooth = (eps + M)**(-gain)
    return (spec * smooth + bias)**power - bias**power

def get_mel_filterbanks(num_banks, fmin, fmax, f_vec, dtype=np.float32):
    '''
    An arguably better version of librosa's melfilterbanks wherein issues with "hard snapping" are avoided. Works with
    an existing vector of frequency bins, as returned from signal.spectrogram(), instead of recalculating them and
    flooring down the bin indices.
    '''

    global CACHE

    # Filterbank already in cache?
    fname = 'mel_' + str(num_banks) + '_' + str(fmin) + '_' + str(fmax)
    if not fname in CACHE:

        # Break frequency and scaling factor
        A = 4581.0
        f_break = 1750.0

        # Convert Hz to mel
        freq_extents_mel = A * np.log10(1 + np.asarray([fmin, fmax], dtype=dtype) / f_break)

        # Compute points evenly spaced in mels
        melpoints = np.linspace(freq_extents_mel[0], freq_extents_mel[1], num_banks + 2, dtype=dtype)

        # Convert mels to Hz
        banks_ends = (f_break * (10 ** (melpoints / A) - 1))

        filterbank = np.zeros([len(f_vec), num_banks], dtype=dtype)
        for bank_idx in range(1, num_banks+1):
            # Points in the first half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx - 1], f_vec <= banks_ends[bank_idx])
            filterbank[mask, bank_idx-1] = (f_vec[mask] - banks_ends[bank_idx - 1]) / \
                (banks_ends[bank_idx] - banks_ends[bank_idx - 1])

            # Points in the second half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx], f_vec <= banks_ends[bank_idx+1])
            filterbank[mask, bank_idx-1] = (banks_ends[bank_idx + 1] - f_vec[mask]) / \
                (banks_ends[bank_idx + 1] - banks_ends[bank_idx])

        # Scale and normalize, so that all the triangles do not have same height and the gain gets adjusted appropriately.
        temp = filterbank.sum(axis=0)
        non_zero_mask = temp > 0
        filterbank[:, non_zero_mask] /= np.expand_dims(temp[non_zero_mask], 0)

        # Save to cache
        CACHE[fname] = (filterbank, banks_ends[1:-1])

    return CACHE[fname][0], CACHE[fname][1]

def spectrogram(sig, rate, shape=(64, 512), win_len=512, fmin=150, fmax=15000, frequency_scale='mel', magnitude_scale='nonlinear', bandpass=True, decompose=False):

    # Compute overlap
    hop_len = int(len(sig) / (shape[1] - 1))
    win_overlap = win_len - hop_len + 2
    #print 'WIN_LEN:', win_len, 'HOP_LEN:', hop_len, 'OVERLAP:', win_overlap

    # Adjust N_FFT?
    if frequency_scale == 'mel':
        n_fft = win_len
    else:
        n_fft = shape[1] * 2

    # Bandpass filter?
    if bandpass:
        sig = applyBandpassFilter(sig, rate, fmin, fmax)

    # Compute spectrogram
    f, t, spec = scipy.signal.spectrogram(sig,
                                          fs=rate,
                                          window=scipy.signal.windows.hann(win_len),
                                          nperseg=win_len,
                                          noverlap=win_overlap,
                                          nfft=n_fft,
                                          detrend=False,
                                          mode='magnitude')

    # Scale frequency?
    if frequency_scale == 'mel':

        # Determine the indices of where to clip the spec
        valid_f_idx_start = f.searchsorted(fmin, side='left')
        valid_f_idx_end = f.searchsorted(fmax, side='right') - 1

        # Get mel filter banks
        mel_filterbank, mel_f = get_mel_filterbanks(shape[0], fmin, fmax, f, dtype=spec.dtype)

        # Clip to non-zero range so that unnecessary multiplications can be avoided
        mel_filterbank = mel_filterbank[valid_f_idx_start:(valid_f_idx_end + 1), :]

        # Clip the spec representation and apply the mel filterbank.
        # Due to the nature of np.dot(), the spec needs to be transposed prior, and reverted after
        spec = np.transpose(spec[valid_f_idx_start:(valid_f_idx_end + 1), :], [1, 0])
        spec = np.dot(spec, mel_filterbank)
        spec = np.transpose(spec, [1, 0])

    # Magnitude transformation
    if magnitude_scale == 'pcen':

        # Convert scale using per-channel energy normalization as proposed by Wang et al., 2017
        # We adjust the parameters for bird voice recognition based on Lostanlen, 2019
        spec = pcen(spec, rate, hop_len)

    elif magnitude_scale == 'log':

        # Convert power spec to dB scale (compute dB relative to peak power)
        spec = spec ** 2
        spec = 10.0 * np.log10(np.maximum(1e-10, spec) / np.max(spec))
        spec = np.maximum(spec, spec.max() - 100) # top_db = 100

    elif magnitude_scale == 'nonlinear':

        # Convert magnitudes using nonlinearity as proposed by Schlüter, 2018
        a = -1.2 # Higher values yield better noise suppression
        s = 1.0 / (1.0 + np.exp(-a))
        spec = spec ** s

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    spec = spec[::-1, ...]

    # Trim to desired shape if too large
    spec = spec[:shape[0], :shape[1]]

    # Normalize values between 0 and 1
    spec -= spec.min()
    if not spec.max() == 0:
        spec /= spec.max()
    else:
        spec = np.clip(spec, 0, 1)

    return spec

def get_spec(sig, rate, spec_type='melspec', **kwargs):

    if spec_type.lower()== 'melspec':
        return spectrogram(sig, rate, frequency_scale='mel', **kwargs)
    else:
        return spectrogram(sig, rate, frequency_scale='linear', **kwargs)

def splitSignal(sig, rate, seconds, overlap, minlen):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break

        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, noise(split, (int(rate * seconds) - len(split)), 0.5)))

        sig_splits.append(split)

    return sig_splits

def specsFromSignal(sig, rate, seconds, overlap, minlen, **kwargs):

    # Split signal in consecutive chunks with overlap
    sig_splits = splitSignal(sig, rate, seconds, overlap, minlen)

    # Extract specs for every sig split
    for sig in sig_splits:

        # Get spec for signal chunk
        spec = get_spec(sig, rate, **kwargs)

        yield spec

def specsFromFile(path, rate, offset=0.0, duration=None, **kwargs):

    # Open file
    sig, rate = openAudioFile(path, rate, offset, duration)

    # Yield all specs for file
    for spec in specsFromSignal(sig, rate, **kwargs):
        yield spec






###### end audio.py ############

##### from BirdNET model.py#########


NONLINEARITY = 'relu'
FILTERS = [8, 16, 32, 64, 128]
KERNEL_SIZES = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]
RESNET_K = 4
RESNET_N = 3

def loadSnapshot(path):

    log.p(('LOADING SNAPSHOT', path.split(os.sep)[-1], '...'), new_line=False)

    with open(path, 'rb') as f:
        try:
            model = pickle.load(f, encoding='latin1')
        except:
            model = pickle.load(f)

    cfg.setModelSettings(model)

    log.p('DONE!')

    return model

def loadParams(net, params):

    log.p('IMPORTING MODEL PARAMS...', new_line=False)

    l.set_all_param_values(net, params)

    log.p('DONE!')

    return net

def logmeanexp(x, axis=None, keepdims=False, sharpness=5):
    # in between maximum (high sharpness) and mean (low sharpness)
    # https://arxiv.org/abs/1411.6228, Eq. 6
    # return T.log(T.mean(T.exp(sharpness * x), axis, keepdims=keepdims)) / sharpness
    # more stable version (Theano can only stabilize the plain logsumexp)
    xmax = T.max(x, axis, keepdims=True)
    xmax2 = T.max(x, axis, keepdims=keepdims)
    x = sharpness * (x - xmax)
    y = T.log(T.mean(T.exp(x), axis, keepdims=keepdims))
    y = y / sharpness + xmax2
    return y

def resblock(net_in, filters, kernel_size, stride=1, preactivated=True, block_id=1, name=''):

    # Show input shape
    #log.p(("\t\t" + name + " IN SHAPE:", l.get_output_shape(net_in)), new_line=False)

    # Pre-activation
    if block_id > 1:
        net_pre = l.NonlinearityLayer(net_in, nonlinearity=nl.rectify)
    else:
        net_pre = net_in

    # Pre-activated shortcut?
    if preactivated:
        net_in = net_pre

    # Bottleneck Convolution
    if stride > 1:
        net_pre = l.batch_norm(l.Conv2DLayer(net_pre,
                                            num_filters=l.get_output_shape(net_pre)[1],
                                            filter_size=1,
                                            pad='same',
                                            stride=1,
                                            nonlinearity=nl.rectify))

    # First Convolution
    net = l.batch_norm(l.Conv2DLayer(net_pre,
                                   num_filters=l.get_output_shape(net_pre)[1],
                                   filter_size=kernel_size,
                                   pad='same',
                                   stride=1,
                                   nonlinearity=nl.rectify))

    # Pooling layer
    #
    if stride > 1:
        net = l.MaxPool2DLayer(net, pool_size=(stride, stride))

    # Dropout Layer
    net = l.DropoutLayer(net)

    # Second Convolution
    net = l.batch_norm(l.Conv2DLayer(net,
                        num_filters=filters,
                        filter_size=kernel_size,
                        pad='same',
                        stride=1,
                        nonlinearity=None))

    # Shortcut Layer
    if not l.get_output_shape(net) == l.get_output_shape(net_in):

        # Average pooling
        shortcut = l.Pool2DLayer(net_in, pool_size=(stride, stride), stride=stride, mode='average_exc_pad')

        # Shortcut convolution
        shortcut = l.batch_norm(l.Conv2DLayer(shortcut,
                                 num_filters=filters,
                                 filter_size=1,
                                 pad='same',
                                 stride=1,
                                 nonlinearity=None))

    else:

        # Shortcut = input
        shortcut = net_in

    # Merge Layer
    out = l.ElemwiseSumLayer([net, shortcut])

    # Show output shape
    #log.p(("OUT SHAPE:", l.get_output_shape(out), "LAYER:", len(l.get_all_layers(out)) - 1))

    return out

def classificationBranch(net, kernel_size):

    # Post Convolution
    branch = l.batch_norm(l.Conv2DLayer(net,
                        num_filters=int(FILTERS[-1] * RESNET_K),
                        filter_size=kernel_size,
                        nonlinearity=nl.rectify))

    #log.p(("\t\tPOST  CONV SHAPE:", l.get_output_shape(branch), "LAYER:", len(l.get_all_layers(branch)) - 1))

    # Dropout Layer
    branch = l.DropoutLayer(branch)

    # Dense Convolution
    branch = l.batch_norm(l.Conv2DLayer(branch,
                        num_filters=int(FILTERS[-1] * RESNET_K * 2),
                        filter_size=1,
                        nonlinearity=nl.rectify))

    #log.p(("\t\tDENSE CONV SHAPE:", l.get_output_shape(branch), "LAYER:", len(l.get_all_layers(branch)) - 1))

    # Dropout Layer
    branch = l.DropoutLayer(branch)

    # Class Convolution
    branch = l.Conv2DLayer(branch,
                        num_filters=len(cfg.CLASSES),
                        filter_size=1,
                        nonlinearity=None)
    return branch

def buildNet():
    '''the main building of the CNN model'''

    log.p('BUILDING BirdNET MODEL...', new_line=False)
    # print('BUILDING BirdNET MODEL...')

    # Input layer for images
    net = l.InputLayer((None, cfg.IM_DIM, cfg.IM_SIZE[1], cfg.IM_SIZE[0]))

    # Pre-processing stage
    #log.p(("\tPRE-PROCESSING STAGE:"))
    net = l.batch_norm(l.Conv2DLayer(net,
                    num_filters=int(FILTERS[0] * RESNET_K),
                    filter_size=(5, 5),
                    pad='same',
                    nonlinearity=nl.rectify))

    #log.p(("\t\tFIRST  CONV OUT SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net)) - 1))

    # Max pooling
    net = l.MaxPool2DLayer(net, pool_size=(1, 2))
    #log.p(("\t\tPRE-MAXPOOL OUT SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net)) - 1))

    # Residual Stacks
    for i in range(1, len(FILTERS)):
        #log.p(("\tRES STACK", i, ':'))
        net = resblock(net,
                       filters=int(FILTERS[i] * RESNET_K),
                       kernel_size=KERNEL_SIZES[i],
                       stride=2,
                       preactivated=True,
                       block_id=i,
                       name='BLOCK ' + str(i) + '-1')

        for j in range(1, RESNET_N):
            net = resblock(net,
                           filters=int(FILTERS[i] * RESNET_K),
                           kernel_size=KERNEL_SIZES[i],
                           preactivated=False,
                           block_id=i+j,
                           name='BLOCK ' + str(i) + '-' + str(j + 1))

    # Post Activation
    net = l.batch_norm(net)
    net = l.NonlinearityLayer(net, nonlinearity=nl.rectify)

    # Classification branch
    #log.p(("\tCLASS BRANCH:"))
    net = classificationBranch(net,  (4, 10))
    #log.p(("\t\tBRANCH OUT SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net)) - 1))

    # Pooling
    net = l.GlobalPoolLayer(net, pool_function=logmeanexp)
    #log.p(("\tGLOBAL POOLING SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net)) - 1))

    # Sigmoid output
    net = l.NonlinearityLayer(net, nonlinearity=nl.sigmoid)

    #log.p(("\tFINAL NET OUT SHAPE:", l.get_output_shape(net), "LAYER:", len(l.get_all_layers(net))))
    log.p("DONE!")
    print("DONE!")

    # Model stats
    #log.p(("MODEL HAS", (sum(hasattr(layer, 'W') for layer in l.get_all_layers(net))), "WEIGHTED LAYERS"))
    #log.p(("MODEL HAS", l.count_params(net), "PARAMS"))

    return net

def test_function(net, layer_index=-1):

    log.p('COMPILING THEANO TEST FUNCTION FUNCTION...', new_line=False)

    prediction = l.get_output(l.get_all_layers(net)[layer_index], deterministic=True)
    test_function = theano.function([l.get_all_layers(net)[0].input_var], prediction, allow_input_downcast=True)

    log.p('DONE!')

    return test_function

def prepareInput(spec):

    # ConvNet inputs in Theano are 4D-vectors: (batch size, channels, height, width)

    # Add axis if grayscale image
    if len(spec.shape) == 2:
        spec = spec[:, :, np.newaxis]

    # Transpose axis, channels = axis 0
    spec = np.transpose(spec, (2, 0, 1))

    # Add new dimension
    spec = np.expand_dims(spec, 0)

    return spec

def flat_sigmoid(x, sensitivity=-1):
    return 1 / (1.0 + np.exp(sensitivity * x))

def predictionPooling(p, sensitivity=-1, mode='avg'):

    # Apply sigmoid function
    p = flat_sigmoid(p, sensitivity)

    # Mean exponential pooling for monophonic recordings
    if mode == 'mexp':
        p_pool = np.mean((p * 2.0) ** 2, axis=0)

    # Simple average pooling
    else:
        p_pool = np.mean(p, axis=0)

    p_pool[p_pool > 1.0] = 1.0

    return p_pool

def predict(spec_batch, test_function):

    # Prediction
    prediction = test_function(spec_batch)

    # Prediction pooling
    p_pool = predictionPooling(prediction, cfg.SENSITIVITY)

    # Get label and scores for pooled predictions
    p_labels = {}
    for i in range(p_pool.shape[0]):
        label = cfg.CLASSES[i]
        if cfg.CLASSES[i] in cfg.WHITE_LIST:
            p_labels[label] = p_pool[i]
        else:
            p_labels[label] = 0.0

    # Sort by score
    p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

    return p_sorted, p_pool

######## end model.py from BirdNET ###########






if __name__ == '__main__':

    # create configs
    cfg = setModelSettings(EBIRD_MDATA)
    print(cfg) # none... well at least it didnt fail... how do we get things like cfg.CLASSES[i] ? cfg.IM_DIM and cfg.IM_SIZE[0] ?


    # load data
    sounds1_txt = pd.read_csv('../example/Soundscape_1.BirdNET.selections.txt')
    # print(sounds1_txt)   # works but probably not using it right

    sig, rate = openAudioFile('../example/Soundscape_1.wav') # probably am supposed to loop this? how to break up the multiple labels and soundbites within the one large wav?
    spec = spectrogram(sig, rate)
    print(spec)
    # probably need to do this for each soundbite, and then each spec gets added to make a spec_batch? whih then is used below in the model?
    # spec_batch = ?


    # net = buildNet() # model... fails bc we don't have cfg.IM_DIM ...
    # test_function = test_function(net)
    # p_sorted, p_pool = predict(spec_batch, test_function)
    # print(p_sorted, p_pool)
