import numpy as np
import pandas as pd
import pickle
import os
import datetime
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking, Bidirectional, GlobalAvgPool1D, GlobalMaxPool1D, Conv1D, TimeDistributed, Input, Concatenate, GRU, dot, multiply, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import scipy.stats as stats
import seaborn as sns
# from scipy.stats import pearsonr
# from scipy.stats import spearmanr
from scipy.stats import ttest_ind

with open('../input/bias-in-ec/audio_featDict.pkl', 'rb') as f:
    audio_featDict=pickle.load(f)

with open('../input/bias-in-ec/audio_featDictMark2.pkl', 'rb') as f:
    audio_featDictMark2=pickle.load(f)

with open('../input/bias-in-ec/genders.pkl', 'rb') as f:
    genders=pickle.load(f)

df= pd.read_csv("../input/bias-in-ec/full_stock_data.csv")

error=[]
error_text=[]

def ModifyData(df, genders = None):
    X=[]
    y_3days_male=[]
    y_7days_male=[]
    y_15days_male=[]
    y_30days_male=[]
    y_3days_female=[]
    y_7days_female=[]
    y_15days_female=[]
    y_30days_female=[]

    if not genders is None:
        print('Got genders --', len(genders))
        X_male=[]
        X_female = []

    for index,row in df.iterrows():
        lstm_matrix_temp = np.zeros((520, 26), dtype=np.float64)
        i=0

        try:
            speaker_list=list(audio_featDict[row['text_file_name']])
            speaker_list=sorted(speaker_list, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
            for sent in speaker_list:
                lstm_matrix_temp[i, :]=audio_featDict[row['text_file_name']][sent]+audio_featDictMark2[row['text_file_name']][sent]
                i+=1
            X.append(lstm_matrix_temp)
            if not genders is None:
                if genders[row['text_file_name']] == 'M':
                    X_male.append(lstm_matrix_temp)
                elif genders[row['text_file_name']] == 'F':
                    X_female.append(lstm_matrix_temp)

        except:
            Padded=np.zeros((520, 26), dtype=np.float64)
            X.append(Padded)
            if not genders is None:
                X_male.append(Padded)
                X_female.append(Padded)
            error.append(row['text_file_name'][:-9])

        if row['text_file_name'] != 'DTE Energy Co._20170726' and genders != None:
            if genders[row['text_file_name']] == 'M':
                y_3days_male.append(float(row['future_3']))
                y_7days_male.append(float(row['future_7']))
                y_15days_male.append(float(row['future_15']))
                y_30days_male.append(float(row['future_30']))
            elif genders[row['text_file_name']] == 'F':
                y_3days_female.append(float(row['future_3']))
                y_7days_female.append(float(row['future_7']))
                y_15days_female.append(float(row['future_15']))
                y_30days_female.append(float(row['future_30']))

    y_3days_male=np.array(y_3days_male)
    y_3days_female=np.array(y_3days_female)
    y_7days_male=np.array(y_7days_male)
    y_7days_female=np.array(y_7days_female)
    y_15days_male=np.array(y_15days_male)
    y_15days_female=np.array(y_15days_female)
    y_30days_male=np.array(y_30days_male)
    y_30days_female=np.array(y_30days_female)

    X=np.array(X)

    if not genders is None:
        X_male=np.array(X_male)
        X_female=np.array(X_female)

    if not genders is None:
        return X,X_male,X_female, y_3days_male,y_3days_female,y_7days_male,y_7days_female,y_15days_male,y_15days_female,y_30days_male,y_30days_female
    else:
        return X

X,X_male,X_female, y_3days_male,y_3days_female,y_7days_male,y_7days_female,y_15days_male,y_15days_female,y_30days_male,y_30days_female=ModifyData(df, genders)

print(X.shape, X_male.shape, X_female.shape)
print(y_3days_male.shape,y_3days_female.shape,y_7days_male.shape,y_7days_female.shape,y_15days_male.shape,y_15days_female.shape,y_30days_male.shape,y_30days_female.shape)

arrays = [X,X_male,X_female]

for arr in arrays:
    inds = np.where(np.isnan(arr))
    print('Number of nans',len(inds[0]))
    if len(inds[0])>0:
        for i in range(len(inds[0])):
            row_mean = np.nanmean(arr[inds[0][i],:,inds[2][i]], axis=0)
            arr[inds[0][i],inds[1][i],inds[2][i]] = row_mean

print(error, error_text)

feature_names = ['Mean F0', 'Stdev F0', 'Hnr', 'Local Jitter', 'Local Absolute Jitter', 'Rap Jitter', 'Ppq5 Jitter', 'Ddp Jitter', 'Local Shimmer', 'Localdb Shimmer', 'Apq3 Shimmer', 'Aqpq5 Shimmer', 'Apq11 Shimmer', 'Dda Shimmer', 'N Pulses', 'N Periods', 'Degree Of Voice Breaks', 'Mean Intensity', 'Sd Energy', 'Max Intensity', 'Min Intensity', 'Max Pitch', 'Min Pitch', 'Voiced Frames', 'Voiced To Total Ratio', 'Voiced To Unvoiced Ratio']

count =0

for i in range(X_male.shape[-1]):
    male_feat = X_male[:,:,i]
    male_avg = np.mean(male_feat, axis =1)

    female_feat = X_female[:,:,i]
    female_avg = np.mean(female_feat, axis =1)

    ## Density plots
    # sns.distplot(male_avg, kde=True, kde_kws={"shade": True}, hist=False, color='#0F84F3')
    # sns.distplot(female_avg, kde=True, kde_kws={"shade": True}, hist=False, color='#F760EE')
    # plt.xlabel(feature_names[i])
    # plt.ylabel('Density')
    # plt.show()
    # plt.savefig(feature_names[i]+'_density.png')

    t, p = ttest_ind(male_avg, female_avg)
    t, p = ttest_ind(male_flatten, female_flatten)
    if p < 0.05 :
        count+=1
    print(feature_names[i], ' : ', p,t)
print('Count of statistically significant different features : ', count)
    # plt.title(feature_names[i])
