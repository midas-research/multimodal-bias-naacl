import numpy as np
import pandas as pd
import os
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
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import tensorflow as tf

with open('./data/audio_featDict.pkl', 'rb') as f:
    audio_featDict=pickle.load(f)

with open('./data/audio_featDictMark2.pkl', 'rb') as f:
    audio_featDictMark2=pickle.load(f)

## add finbert embeddings here --> change path
with open('./data/finbert_earnings.pkl', 'rb') as f:
    text_dict=pickle.load(f)

with open('./data/genders.pkl', 'rb') as f:
    genders=pickle.load(f)

traindf= pd.read_csv("./data/train_split3.csv")
testdf=pd.read_csv("./data/test_split3.csv")
valdf=pd.read_csv("./data/val_split3.csv")

error=[]
error_text=[]

print(len(text_dict))

def ModifyData(df,text_dict, genders = None):
    X=[]  #Audio
    X_text=[]   #Text
    y_3days=[]
    y_7days=[]
    y_15days=[]
    y_30days=[]

    if not genders is None:
        print('Got genders --', len(genders))
        X_male=[]
        X_female = []
        X_text_male = []
        X_text_female = []
        y_7days_male = []
        y_7days_female = []

    for index,row in df.iterrows():


        try:
            X_text.append(text_dict[row['text_file_name']])
            if not genders is None:
                if index == 0:
                    print(genders[row['text_file_name']])
                if genders[row['text_file_name']] == 'M':
                    X_text_male.append(text_dict[row['text_file_name']])

                elif genders[row['text_file_name']] == 'F':
                    X_text_female.append(text_dict[row['text_file_name']])
        except:
            error_text.append(row['text_file_name'][:-9])

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
            X_male.append(Padded)
#             if not genders is None:
#                 if genders[row['text_file_name']] == 'M':
#                     X_male.append(Padded)
#                 elif genders[row['text_file_name']] == 'F':
#                     X_female.append(Padded)
            error.append(row['text_file_name'][:-9])


        try:
            y_3days.append(float(row['future_3']))
            if not genders is None:
                if genders[row['text_file_name']] == 'M':
                    y_7days_male.append(float(row['future_7']))
                elif genders[row['text_file_name']] == 'F':
                    y_7days_female.append(float(row['future_7']))
            y_7days.append(float(row['future_7']))
            y_15days.append(float(row['future_15']))
            y_30days.append(float(row['future_30']))

        except:
            y_7days.append(float(row['future_7']))
            y_7days_male.append(float(row['future_7']))

    X=np.array(X)
    X_text=np.array(X_text)
    y_3days=np.array(y_3days)

    if not genders is None:
        X_male=np.array(X_male)
        X_text_male=np.array(X_text_male)
        y_7days_male=np.array(y_7days_male)
        X_female=np.array(X_female)
        X_text_female=np.array(X_text_female)
        y_7days_female=np.array(y_7days_female)

    y_7days=np.array(y_7days)
    y_15days=np.array(y_15days)
    y_30days=np.array(y_30days)

    X=np.nan_to_num(X)

    if not genders is None:
        return X,X_text,y_3days,y_7days,y_15days,y_30days, X_male,X_text_male,y_7days_male, X_female,X_text_female,y_7days_female
    else:
        return X,X_text,y_3days,y_7days,y_15days,y_30days


X_train_audio,X_train_text,y_train3days, y_train7days, y_train15days, y_train30days,X_train_audio_male,X_train_text_male,y_train_7days_male,X_train_audio_female,X_train_text_female,y_train_7days_female=ModifyData(traindf,text_dict,genders)

X_test_audio,X_test_text,y_test3days, y_test7days, y_test15days, y_test30days, X_test_audio_male,X_test_text_male,y_test_7days_male, X_test_audio_female,X_test_text_female,y_test_7days_female=ModifyData(testdf,text_dict, genders)

X_val_audio,X_val_text,y_val3days, y_val7days, y_val15days, y_val30days=ModifyData(valdf,text_dict)

print(X_train_audio.shape, X_train_text.shape)
input_audio_shape = (X_train_audio.shape[1], X_train_audio.shape[2])
input_text_shape = (X_train_text.shape[1],X_train_text.shape[2])

# print(X_train_audio.shape,X_train_text.shape,y_train3days.shape, X_train_text_male.shape,X_train_text_female.shape,X_train_audio_male.shape,X_train_audio_female.shape,y_train_3days_male.shape,y_train_3days_female.shape)
# print(X_test_audio.shape,X_test_text.shape,y_test3days.shape, y_test7days.shape, y_test15days.shape, y_test30days.shape, X_test_audio_male.shape,X_test_text_male.shape,y_test_3days_male.shape, X_test_audio_female.shape,X_test_text_female.shape,y_test_3days_female.shape)

###Audio Model

#print(tf.test.gpu_device_name())
def train_audio(X_train_audio_n,turn, duration,y_train_new, y_val, y_test, y_test_male, y_test_female, dropout, units, batch_size, epochs, learning_rate):
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # model definiion starts here
    with tpu_strategy.scope():
        sequence_input1 = Input(shape=input_audio_shape, dtype='float32')

        x= (Bidirectional(LSTM(units=100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))(sequence_input1)
        x = Dropout(0.5)(x)

        x = TimeDistributed(Dense(100,activation='relu'))(x)
        x = Bidirectional(LSTM(units=100,dropout=0, recurrent_dropout=0,activation='tanh'))(x)

        x = Dense(128,activation='relu', name='features1')(x)
        x = Dense(64,activation='relu', name='features2')(x)
        main_output = Dense(1,activation='linear', name='main_output')(x)
        model = Model(inputs=sequence_input1, outputs=main_output)
        adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile( optimizer=adam,loss='mean_squared_error')

    print(model.summary())
    print(X_train_audio_n.shape)
    print(y_train_new.shape)

    history=model.fit(
        X_train_audio_n,
        y_train_new,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_audio , y_val)
        )

    train_loss = model.evaluate(X_train_audio_n,y_train_new,batch_size=batch_size)


    test_loss = model.evaluate(X_test_audio,y_test,batch_size=batch_size)
    test_loss_male = model.evaluate(X_test_audio_male,y_test_male,batch_size=batch_size)
    test_loss_female = model.evaluate(X_test_audio_female,y_test_female,batch_size=batch_size)


    print("Train loss  : {train_loss}".format(train_loss = train_loss))
    print("Test loss : {test_loss}".format(test_loss = test_loss))
    print("Test loss male: {test_loss_male}".format(test_loss_male = test_loss_male))
    print("Test loss female: {test_loss_female}".format(test_loss_female = test_loss_female))


    print()
    y_pred = model.predict(X_test_audio)
    y_pred_m = model.predict(X_test_audio_male)
    y_pred_f = model.predict(X_test_audio_female)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    save_path = "epochs="+str(epochs)+"_dropout="+str(dropout)+"_learning-rate"+str(learning_rate)+"_units="+str(units)
    save_pkl="y_pred_"+str(duration)+"audio"+str(turn)+".pkl"
    save_pkl_m="y_pred_"+str(duration)+"audio_m"+str(turn)+".pkl"
    save_pkl_f="y_pred_"+str(duration)+"audio_f"+str(turn)+".pkl"


    with open(save_pkl,'wb') as f:
        pickle.dump(y_pred,f)
    with open(save_pkl_m,'wb') as f:
        pickle.dump(y_pred_m,f)
    with open(save_pkl_f,'wb') as f:
        pickle.dump(y_pred_f,f)

    model.save(save_path+"audio_model"+str(turn)+".h5")
    plt.savefig(save_path+"audio"+str(turn)+".png")
    plt.show()
    plt.close()
    return test_loss, test_loss_male, test_loss_female,train_loss

def train_text(X_train_text_n,turn, duration,y_train_n, y_val, y_test, y_test_male, y_test_female, dropout, units, batch_size, epochs, learning_rate):

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # model definiion starts here
    with tpu_strategy.scope():

        sequence_input2 = Input(shape=input_text_shape, dtype='float32')

        y= (Bidirectional(LSTM(units=100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))(sequence_input2)
        y = Dropout(0.5)(y)

        y = TimeDistributed(Dense(100,activation='relu'))(y)
        y = Bidirectional(LSTM(units=100,dropout=0, recurrent_dropout=0,activation='tanh'))(y)

        x = Dense(128,activation='relu', name='features1')(y)
        x = Dense(64,activation='relu', name='features2')(x)
        main_output = Dense(1,activation='linear', name='main_output')(x)
        model = Model(inputs=sequence_input2, outputs=main_output)
        adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile( optimizer=adam,loss='mean_squared_error')

    print(model.summary())

    history=model.fit(
        X_train_text_n,
        y_train_n,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_text , y_val))


    train_loss = model.evaluate(X_train_text_n,y_train_n,batch_size=batch_size)

    test_loss = model.evaluate(X_test_text,y_test,batch_size=batch_size)
    test_loss_male = model.evaluate(X_test_text_male,y_test_male,batch_size=batch_size)
    test_loss_female = model.evaluate(X_test_text_female,y_test_female,batch_size=batch_size)




    print()
    print("Train loss  : {train_loss}".format(train_loss = train_loss))
    print("Test loss : {test_loss}".format(test_loss = test_loss))
    print("Test loss male: {test_loss_male}".format(test_loss_male = test_loss_male))
    print("Test loss female: {test_loss_female}".format(test_loss_female = test_loss_female))


    print()
    y_pred = model.predict(X_test_text)
    y_pred_m = model.predict(X_test_text_male)
    y_pred_f = model.predict(X_test_text_female)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    save_path = "epochs="+str(epochs)+"_dropout="+str(dropout)+"_learning-rate"+str(learning_rate)+"_units="+str(units)
    save_pkl="y_pred_"+str(duration)+"text"+str(turn)+".pkl"
    save_pkl_m="y_pred_"+str(duration)+"text_m"+str(turn)+".pkl"
    save_pkl_f="y_pred_"+str(duration)+"text_f"+str(turn)+".pkl"


    with open(save_pkl,'wb') as f:
        pickle.dump(y_pred,f)
    with open(save_pkl_m,'wb') as f:
        pickle.dump(y_pred_m,f)
    with open(save_pkl_f,'wb') as f:
        pickle.dump(y_pred_f,f)

    model.save(save_path+"text_model"+str(turn)+".h5")
    plt.savefig(save_path+"text"+str(turn)+".png")
    plt.show()
    plt.close()
    return test_loss, test_loss_male, test_loss_female,train_loss


def train_at(X_train_audio_n,X_train_text_n,turn, duration,y_train_new, y_val, y_test, y_test_male, y_test_female, dropout, units, batch_size, epochs, learning_rate):

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # model definiion starts here
    with tpu_strategy.scope():
        sequence_input1 = Input(shape=input_audio_shape, dtype='float32')

        x= (Bidirectional(LSTM(units=100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))(sequence_input1)
        x = Dropout(0.5)(x)

        x = TimeDistributed(Dense(100,activation='relu'))(x)
        x = Bidirectional(LSTM(units=100,dropout=0, recurrent_dropout=0,activation='tanh'))(x)

        sequence_input2 = Input(shape=input_text_shape, dtype='float32')

        y= (Bidirectional(LSTM(units=100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))(sequence_input2)
        y = Dropout(0.5)(y)

        y = TimeDistributed(Dense(100,activation='relu'))(y)
        y = Bidirectional(LSTM(units=100,dropout=0, recurrent_dropout=0,activation='tanh'))(y)

        x = concatenate([x, y])

        x = Dense(128,activation='relu', name='features1')(x)
        x = Dense(64,activation='relu', name='features2')(x)
        main_output = Dense(1,activation='linear', name='main_output')(x)
        model = Model(inputs=[sequence_input1, sequence_input2], outputs=main_output)
        adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)


        model.compile( optimizer=adam,loss='mean_squared_error')

    print(model.summary())

    history=model.fit(
        [X_train_audio_n, X_train_text_n],
        y_train_new,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_val_audio, X_val_text] , y_val))


    test_loss = model.evaluate([X_test_audio, X_test_text],y_test,batch_size=batch_size)
    test_loss_male = model.evaluate([X_test_audio_male, X_test_text_male],y_test_male,batch_size=batch_size)
    test_loss_female = model.evaluate([X_test_audio_female, X_test_text_female],y_test_female,batch_size=batch_size)

    train_loss = model.evaluate([X_train_audio_n, X_train_text_n],y_train_new,batch_size=batch_size)

    print()
    print("Train loss  : {train_loss}".format(train_loss = train_loss))
    print("Test loss : {test_loss}".format(test_loss = test_loss))
    print("Test loss male: {test_loss_male}".format(test_loss_male = test_loss_male))
    print("Test loss female: {test_loss_female}".format(test_loss_female = test_loss_female))


    print()
    y_pred = model.predict([X_test_audio, X_test_text])
    y_pred_m = model.predict([X_test_audio_male, X_test_text_male])
    y_pred_f = model.predict([X_test_audio_female, X_test_text_female])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    save_path = "epochs="+str(epochs)+"_dropout="+str(dropout)+"_learning-rate"+str(learning_rate)+"_units="+str(units)
    save_pkl="y_pred_"+str(duration)+"audio_text"+str(turn)+".pkl"
    save_pkl_m="y_pred_"+str(duration)+"audio_text_m"+str(turn)+".pkl"
    save_pkl_f="y_pred_"+str(duration)+"audio_text_f"+str(turn)+".pkl"


    with open(save_pkl,'wb') as f:
        pickle.dump(y_pred,f)
    with open(save_pkl_m,'wb') as f:
        pickle.dump(y_pred_m,f)
    with open(save_pkl_f,'wb') as f:
        pickle.dump(y_pred_f,f)

    model.save(save_path+"audio_text_model"+str(turn)+".h5")
    plt.savefig(save_path+"audio_text"+str(turn)+".png")
    plt.show()
    plt.close()
    return test_loss, test_loss_male, test_loss_female,train_loss

test_arr = [X_test_audio,X_test_text,X_test_text_male,X_test_text_female,X_test_audio_male,X_test_audio_female]
train_arr = [X_train_audio,X_train_text,y_train7days, X_train_text_male,X_train_text_female,X_train_audio_male,X_train_audio_female,y_train_7days_male,y_train_7days_female]

# for treating nans


for arr in test_arr:
    inds = np.where(np.isnan(arr))
    print('Number of nans',len(inds[0]))
    if len(inds[0])>0:
        for i in range(len(inds[0])):
            row_mean = np.nanmean(arr[inds[0][i],:,inds[2][i]], axis=0)
            arr[inds[0][i],inds[1][i],inds[2][i]] = row_mean

for arr in train_arr:
    inds = np.where(np.isnan(arr))
    print('Number of nans',len(inds[0]))
    if len(inds[0])>0:
        for i in range(len(inds[0])):
            row_mean = np.nanmean(arr[inds[0][i],:,inds[2][i]], axis=0)
            arr[inds[0][i],inds[1][i],inds[2][i]] = row_mean

ratio = X_train_audio_male.shape[0]/X_train_audio_female.shape[0]
num_male = X_train_audio_male.shape[0]
male_len_list = []

for i in range(1,int(ratio)+1):
    test_male_len = (1/i)*num_male
    test_male_len = int(test_male_len)
    male_len_list.append(test_male_len)


male_len_list = list(reversed(male_len_list))
male_len_list

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#For audio

mse_ = []
mse_m_ = []
mse_f_ = []

for j in range(5):
    mse = []
    mse_m = []
    mse_f = []
    for i in range(len(male_len_list)):
        print(i)

        X_train_audio_male_new = X_train_audio_male[:male_len_list[i]]
        X_train_text_male_new = X_train_text_male[:male_len_list[i]]
        y_train_male_new = y_train_7days_male[:male_len_list[i]]

        print(X_train_audio_male_new.shape)


        X_train_audio_new = np.concatenate([X_train_audio_male_new,X_train_audio_female])
        X_train_text_new = np.concatenate([X_train_text_male_new,X_train_text_female])
        y_train_new = np.concatenate([y_train_male_new,y_train_7days_female])

        print(X_train_audio_new.shape)
        print(y_train_new.shape)

        X_train_audio_new,y_train_new = unison_shuffled_copies(X_train_audio_new,y_train_new)


        mse_audio, mse_m_audio, mse_f_audio,train_loss = train_audio(X_train_audio_new,i, 7,y_train_new, y_val7days, y_test7days, y_test_7days_male, y_test_7days_female, dropout=0.5, units=100, batch_size=16, epochs=20, learning_rate=0.001)

        mse.append(mse_audio)
        mse_m.append(mse_m_audio)
        mse_f.append(mse_f_audio)
    mse_.append(mse)
    mse_m.append(mse_m)
    mse_f.append(mse_f)

mses = []

for arr in mse_:
    mses.append(arr)

for arr in mse_m_:
    mses.append(arr)

for arr in mse_f_:
    mses.append(arr)

mse_df=pd.DataFrame(mses)
mse_df.to_csv('./MSE_audio.csv')



# print('The mean and standard deviation for the MSEs for AT are:')
# print('Complete test set :')
# print('Mean', np.mean(np.array(mse_)))
# print('Std dev : ', np.std(np.array(mse_)))
# print('Male only :')
# print('Mean', np.mean(np.array(mse_m_)))
# print('Std dev : ', np.std(np.array(mse_m_)))
# print('Female only :')
# print('Mean', np.mean(np.array(mse_f_)))
# print('Std dev : ', np.std(np.array(mse_f_)))

# #For text

# mse_ = []
# mse_m_ = []
# mse_f_ = []


# for i in range(len(male_len_list)):
#     print(i)

#     X_train_audio_male_new = X_train_audio_male[:male_len_list[i]]
#     X_train_text_male_new = X_train_text_male[:male_len_list[i]]
#     y_train_male_new = y_train_audiodays_male[:male_len_list[i]]

#     print(X_train_audio_male_new.shape)


#     X_train_audio_new = np.concatenate([X_train_audio_male_new,X_train_audio_female])
#     X_train_text_new = np.concatenate([X_train_text_male_new,X_train_text_female])
#     y_train_new = np.concatenate([y_train_male_new,y_train_3days_female])

#     print(X_train_audio_new.shape)
#     print(y_train_new.shape)

#     X_train_audio_new,y_train_new = unison_shuffled_copies(X_train_audio_new,y_train_new)

#     mse_text, mse_m_text, mse_f_text,train_loss = train_text(X_train_text_new,i, 3,y_train_new, y_val3days, y_test3days, y_test_3days_male, y_test_3days_female, dropout=0.5, units=100, batch_size=16, epochs=20, learning_rate=0.001)
#     #mse_at, mse_m_at, mse_f_at,train_loss = train_at(i, 3,y_train3days, y_val3days, y_test3days, y_test_3days_male, y_test_3days_female, dropout=0.5, units=100, batch_size=16, epochs=1, learning_rate=0.001)
#     #mse_audio, mse_m_audio, mse_f_audio,train_loss = train_audio(X_tt,i, 3,y_tt, y_val3days, y_test3days, y_test_3days_male, y_test_3days_female, dropout=0.5, units=100, batch_size=16, epochs=1, learning_rate=0.001)

#     mse_.append(mse_text)
#     mse_m_.append(mse_m_text)
#     mse_f_.append(mse_f_text)




# # print('The mean and standard deviation for the MSEs for AT are:')
# # print('Complete test set :')
# # print('Mean', np.mean(np.array(mse_)))
# # print('Std dev : ', np.std(np.array(mse_)))
# # print('Male only :')
# # print('Mean', np.mean(np.array(mse_m_)))
# # print('Std dev : ', np.std(np.array(mse_m_)))
# # print('Female only :')
# # print('Mean', np.mean(np.array(mse_f_)))
# # print('Std dev : ', np.std(np.array(mse_f_)))

# x = ['1:1','2:1','3:1','4:1','5:1','6:1','7:1','8:1']
# xi = list(range(len(x)))
# plt.plot(atm)
# plt.plot(atf)
# plt.title('Gender wise Loss')
# plt.ylabel('Loss')
# plt.xlabel('M:F Ratio')
# plt.xticks(xi, x)
# plt.legend(['Test Loss Male', 'Test Loss Female'], loc='upper left')


# #For at

# mse_ = []
# mse_m_ = []
# mse_f_ = []


# for i in range(len(male_len_list)):
#     print(i)

#     X_train_audio_male_new = X_train_audio_male[:male_len_list[i]]
#     X_train_text_male_new = X_train_text_male[:male_len_list[i]]
#     y_train_male_new = y_train_3days_male[:male_len_list[i]]

#     print(X_train_audio_male_new.shape)


#     X_train_audio_new = np.concatenate([X_train_audio_male_new,X_train_audio_female])
#     X_train_text_new = np.concatenate([X_train_text_male_new,X_train_text_female])
#     y_train_new = np.concatenate([y_train_male_new,y_train_3days_female])

#     print(X_train_audio_new.shape)
#     print(y_train_new.shape)

#     X_train_audio_new,y_train_new = unison_shuffled_copies(X_train_audio_new,y_train_new)

# #     X_tt = X_train_audio[:391]
# #     y_tt =  y_train3days[:391]

#     #mse_audio, mse_m_audio, mse_f_audio,train_loss = train_audio(X_train_audio_new,i, 3,y_train_new, y_val3days, y_test3days, y_test_3days_male, y_test_3days_female, dropout=0.5, units=100, batch_size=16, epochs=20, learning_rate=0.001)
#     mse_at, mse_m_at, mse_f_at,train_loss = train_at(X_train_audio_new,X_train_text_new,i, 3,y_train_new, y_val3days, y_test3days, y_test_3days_male, y_test_3days_female, dropout=0.5, units=100, batch_size=16, epochs=20, learning_rate=0.001)
#     #mse_audio, mse_m_audio, mse_f_audio,train_loss = train_audio(X_tt,i, 3,y_tt, y_val3days, y_test3days, y_test_3days_male, y_test_3days_female, dropout=0.5, units=100, batch_size=16, epochs=1, learning_rate=0.001)

#     mse_.append(mse_at)
#     mse_m_.append(mse_m_at)
#     mse_f_.append(mse_f_at)
