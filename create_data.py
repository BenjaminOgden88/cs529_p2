import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#used for mode selction in chroma
from collections import Counter
import random

from sklearn import preprocessing

#Creates column names for each csv file. Note this wont display anymore due to normaliztion step just replacing
#strings with int values
def make_description(feature_extraction_type : str ,total_features_from_extraction_type : int):
    ft = []
    for i in range(total_features_from_extraction_type):
        ft.append(feature_extraction_type + " " + str(i))
    return ft

# say we have 2d array, where one dim is the coeffs, and the other dim is the coef over time
# this will basically for each coef, bucket its time steps into buckets, mean those, then we
# get one list of something like [coef 0 mean 0,coef 0 mean 1...coef 0 mean b,coef 1 mean 0...coef n mean b]
# NOTE: this was use in out bucketing attempts for each feature extraciton methouds. But each time 
# bucketing was used the data performed poorly, so its currently not being used.
def convert_sd_array_to_list_buckets(two_dim_array : np.array, mode:str,number_buckets: int):
    all_values = []
    #number of buckets for each coef
    #ATTEMPT april 1, lets try to, for each coef, do buckets of all steps, with b buckets,that way we get more coef
    for i in two_dim_array:
        buckets = np.array_split(np.array(i),number_buckets)
        
        for j in buckets:
            mean_given_bucket = None
            if mode == "mean":
                mean_given_bucket = j.mean()
            elif mode == "median":
                mean_given_bucket = np.median(j)
            elif mode == "std":
                mean_given_bucket = np.std(j)
            else:
                vals, counts = np.unique(j, return_counts=True)

                #give list of all values that were highest mode
                mode_value = np.argwhere(counts == np.max(counts))
                #if multi, mean all modes up, return that
                mean_given_bucket = np.mean(mode_value)
            all_values.append(mean_given_bucket)
    return all_values

def MFCC(signal,sr,hop_length,number_mfcc,window_length):
    #MFCC
    ######################################################################################################################################################
    #extract MFCC's,
    #MFCC coeficients contain info about the rate of changes in different spectrum bands
    #spectrum bands are is spectrum with frqecneus 3Hz to 3,000 Ghz.

    #positive cepstral coeff => m,ajority of spectral energfy is conmcentrated in low freq region
    #negative value => most spectral energfy is in high frequency

    #mfcc uses small frames, not entire audio file right away i assume, as the fourier transform acrros wole singla would lose ferquency contours over time.
    #25 ms frames are standard
    #EXAMPLE if frame lengh for 16kHz signal is 0.025(standard frame) * 16000 = 400 SAMPLES wit smaple hop of 160 samples
    
    #extract MFCCS
    mfccs = librosa.feature.mfcc(y=signal,sr=sr,hop_length = 50,n_mfcc=128,win_length = window_length,lifter=200)
    ############################################################
    #Mean and std attempt, best one so far
    mfccs_avg = np.mean(mfccs,axis=1)
    mfccs_std = np.std(mfccs,axis=1)
    return list(mfccs_avg) + list(mfccs_std)
    #########################################


    #Bucket attempt. Tho it sucked#
    #return convert_sd_array_to_list_buckets(mfccs,"mode",400)

def chroma_features(signal,sr,hop_length,window_length):
    ######################################################################################################################################################
    #CHROMA FEATURES
    #this source was very helpful with understanding extracting chroma features
    # https://medium.com/@oluyaled/detecting-musical-key-from-audio-using-chroma-feature-in-python-72850c0ae4b1
    #chroma features help us detect musical key(ie the central pitch that is used through duration of a song)
    #there are 12 major scales in music, C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', and 'B'

    chroma_coeff = librosa.feature.chroma_stft(y=signal,sr=sr,hop_length=50,win_length=window_length)
    #####################################################################################################
    #Means and std attempt. So far best one 
    chroma_means =  np.mean(chroma_coeff,axis = 1)
    chroma_std = np.std(chroma_coeff,axis = 1)
    return list(chroma_means) + list(chroma_std)
    ##########################################################

    ###############################################################################################
    #for this methoud, for each time step, look at all 13 chroma coef,
    #and pick largest one, well say thats the key at that time anad we
    #keep it as a feature.
    #WONT KEEP CHROMA KEY WILL JUST KEEP INDEX AS WE NEED NUMBERS FOR LOGIST
    #REGRESSION NOT CATEGORICAL DATA
    # list_keys = []
    # for j in range(chroma_coeff.shape[1]):
    #     max = chroma_coeff[0][j]
    #     index = 0
    #     for i in range(chroma_coeff.shape[0]):
    #         curr = chroma_coeff[i][j]
    #         if curr > max:
    #             max = curr
    #             index = i
    #     list_keys.append(index)
    # print("list_keys ",len(list_keys))
    # #for some reason somtimes not 1293 size lets force it
    # list_keys = np.array_split(np.array(list_keys),5)
    # return_list = []
    # for i in list_keys:
    #     #return_list.append(np.mean(i))
    #     a = Counter(i)
    #     #basically get mode key found
    #     #TODO possibly looks like Counter only gets first tiwbreaker, ie if index 0 and 1 have same number of occurances
    #     #will always pick 0, however I dont think this will matter has chances that two index get same # occurances is
    #     #very low
    #     #print(a.most_common(1)[0][0])
    #     return_list.append(a.most_common(1)[0][0])
    #     #try with own mode mode
    # #print(return_list)
    # return return_list
    #################################################################################################


    ##############################################################
    #second attempt. Bucket attempt
    #return convert_sd_array_to_list_buckets(chroma_coeff,"mean",4)
    ############################################################


#THIS WAS IGNORED IN FINAL DATASET, we will keep it here however to show what we attempted it
def STFT(signal,hop_length,n_fft_stft,window_length):

    ##########################################################################
    #NOW USING JUST THE STFT
    stft_coeff = librosa.stft(signal,hop_length=hop_length,n_fft = 2048,win_length=window_length)



    #convert from amps to decibls becasue we can limit the range of possible values., otherwise would have millions of possible
    # values
    #https://stackoverflow.com/questions/63347977/what-is-the-conceptual-purpose-of-librosa-amplitude-to-db

    #i believe from formula magnitude [dB] = 20 * Log(sqr(Re^2 + Im^2))
    #https://www.rohde-schwarz.com/us/faq/converting-the-real-and-imaginary-numbers-to-magnitude-in-db-and-phase-in-degrees-faq_78704-30465.html
    decibal_coinvserion_from_stft = librosa.amplitude_to_db(abs(stft_coeff))

    #make 4 buckets for each fft
    return convert_sd_array_to_list_buckets(decibal_coinvserion_from_stft,"mean",4)


def spectral_contrast(signal,sr,hop_length,window_length):
    ########################################################################################################################################################################################
    #SPECTRAL CONTRAST

    #use this line as in example for documentation
    #https://librosa.org/doc/0.10.1/generated/librosa.feature.spectral_contrast.html#librosa-feature-spectral-contrast
    #S = np.abs(librosa.stft(signal,hop_length=hop_length))

    #spectral_contrast should return spec contrast corr to gien octave based frequency, for each row
    #REMOVED S FROM SPECTRAL CONTARAST AND ACTUALLY GOT BETTER EPRFORMANCE????
    spectral_contrast = librosa.feature.spectral_contrast(y=signal,sr=sr,win_length=window_length)
    spec_mean = np.mean(spectral_contrast, axis = 1)
    spec_std = np.std(spectral_contrast, axis = 1)
    return list(spec_mean) + list(spec_std)

#False, meaning more times the signal did NOT change from one sign to zero to the opposite sign
#True, meaning more times the signal did in fact change from one sign to zero to the opposite sign
#returns ratio of falses to total
def zero_crossing_rate(signal):
    ########################################################################################################################################
    #ZERO CROSSING RATE
    #NO HOP LEGNTH FOUND FOR ZERO CROSSINGS
    zero_crossing = librosa.zero_crossings(signal,pad=False)
    zeros_ones_list = np.bincount(zero_crossing)
    return zeros_ones_list[0] / (len(zero_crossing))

#get tempo for audio file
def tempo(signal,sr):
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    #tempo = convert_sd_array_to_list_buckets([tempo],"mean",40)
    return tempo[0]




# Turns all feature extraction features into one list for dataframe creation purposes
def make_one_list_all_data(all_features):
    list_all_row_data = []
    for i in all_features:
        
        if type(i) == np.ndarray or type(i) == list:
            for j in i:
                list_all_row_data.append(j)
        #the given featrue extraction methoud was a single value
        else:
            list_all_row_data.append(i)
    return list_all_row_data



# For each sample. Create data using desired feautre extraciton methouds
def create_row(audio_path,file_id,genre,first_runthrough : bool,sample_rate,hop_length,number_mfcc,n_fft_stft,window_length):
    signal, sr = librosa.load(audio_path,sr=sample_rate,mono=True)
    mfcc_features = MFCC(signal,sr,hop_length,number_mfcc,window_length)
    chroma_features_ = chroma_features(signal,sr,hop_length,window_length)
    spectral_contrast_features = spectral_contrast(signal,sr,hop_length,window_length)
    zcr_feature = zero_crossing_rate(signal)
    tempo_feature = tempo(signal,sample_rate)

    title_list = []
    #if first runthorugh. then we need titles for dataframe
    #NOTE titales get removed during normlization phase. make_desciription was just kept to avoid erros.
    if first_runthrough == True:
        mfcc_title = make_description("MFCC", len(mfcc_features))
        chroma_title = make_description("Chroma", len(chroma_features))
        #stft_title = make_description("STFT", len(average_decibal_from_stft_matrix))
        spectral_title = make_description("Spectral", len(spectral_contrast_features))
        zcr_title = make_description("ZCR", 1)
        tempo_feature_title = make_description("Tempo", 1)

        #depending on what feature exctactions used place here
        title_list = mfcc_title + chroma_title + spectral_title + zcr_title + tempo_feature_title + ["id"]
        if genre != None:
            title_list.append("class")

    #Now use desured attribute values to create the given row for a file, as well as the audio target type(blues,classical, etc), determined
    list_all_row_data = None
    print("extracted audio and turned to row of data from file " + str(audio_path))
    if genre != None:
        list_all_row_data = make_one_list_all_data([mfcc_features,chroma_features_,spectral_contrast_features,zcr_feature,tempo_feature,file_id,genre])
        return list_all_row_data,title_list
    else:
        list_all_row_data = make_one_list_all_data([mfcc_features,chroma_features_,spectral_contrast_features,zcr_feature,tempo_feature,file_id])
        return list_all_row_data,title_list

#train_or_test will either be "test" or "train" data frame building
#go thorugh directory and get all audio files(with .au)
def create_data_frame(rootdir,train_or_test,sample_rate,hop_length,number_mfcc,n_fft_stft,window_length):
    data_frame = None

    first_run_through = True
    given_row = 0
    for subdir,dirs,files in os.walk(rootdir):
        for file in files:
            if train_or_test in subdir and ".au" in file:
                #audio tpye is blues,classical,or country, etc...
                genre = None
                if train_or_test == "train":
                    genre = file.split(".")[0]

                file_id = str(file)
                attr_list = create_row(os.path.join(subdir, file),file_id,genre,first_run_through,sample_rate,hop_length,number_mfcc,n_fft_stft,window_length)
                if first_run_through == True:
                    data_frame = pd.DataFrame(columns=attr_list[1])
                data_frame.loc[given_row] = attr_list[0]
                given_row = given_row + 1
                print("added row to dataframe")
                first_run_through = False

    #Normalize the created data
    list_ignor_in_scaler = ["id","class"]
    if train_or_test == "test":
        list_ignor_in_scaler = ["id"]
    hold_col = []
    for i in list_ignor_in_scaler:
        hold_col.append(data_frame[i])
        del data_frame[i]

    x = data_frame.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)

    #now add back audio id and classes after the normalize was done
    df = df.assign(id=pd.Series(hold_col[0]))
    if train_or_test == "train":
        df["class"] = hold_col[1]

    return df
