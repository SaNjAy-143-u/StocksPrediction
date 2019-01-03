# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.utils import plot_model
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tf_data import TF_Data
import pandas
from keras.models import load_model
from os.path import isfile
import os
from keras.optimizers import Adam,SGD,RMSprop,Adadelta
import pickle
def train(filename, model_name, day='today'):
    # reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')
    # checkpoint = ModelCheckpoint(model_name, monitor='val_acc', mode='auto', verbose=1, save_best_only=True)
    # earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1)
    #df = pandas.read_csv(filename, sep='\t')
    #headlines = df['normalized_headline'].as_matrix()
    top_words = 2000

    data = TF_Data(filename, top_words=top_words)
    pickle.dump(data, open(filename.replace(".csv", ".p"), "wb"))
    # load the dataset but only keep the top n words, zero the rest
    (total_price_train,total_X_train, total_y_train), (price_test,X_test, y_test) = data.load_data(day=day)
    
    # truncate and pad input sequences
    max_review_length = 10000
    total_X_train = sequence.pad_sequences(total_X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32
    #news sentiments
    input1=Input(shape=(max_review_length,))
    embed=Embedding(top_words, embedding_vecor_length)(input1)
    conv=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embed)
    pool1=MaxPooling1D(pool_size=2)(conv)
    lstm_news=LSTM(100,return_sequences=True)(pool1)
    
    #50 stocks as input
    # input2=Input(shape=(50,1,5,))
    # lstm_price=[]
    # for i in range(50):
    #     lstm_price.append(LSTM(100,return_sequences=True)(input2[i]))
    # concat_stocks=concatenate(lstm_price,axis=1)

    #stock price history
    input2=Input(shape=(500,1,))
    lstm_price=LSTM(100,return_sequences=True)(input2)
    #concatenation of both
    concat=concatenate([lstm_news,lstm_price],axis=1)
    final_LSTM=LSTM(100)(concat)
    dense=Dense(32,activation='relu')(final_LSTM)
    final=Dense(8,activation='softmax')(dense)
    model=Model(inputs=[input1,input2],outputs=final)
    optimizer = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if isfile(model_name) and False:
        print('Checkpoint Loaded')
        model = load_model(model_name)
    print(model.summary())
    plot_model(model, to_file='convolutional_neural_network.png')
    # model.fit(X_train, y_train, epochs=3, batch_size=1, callbacks=[checkpoint, earlyStopping, reduceLR])
    start=0
    siz=int(len(total_price_train)/5)
    end=siz
    for i in range(5):
        if isfile(model_name):
            print('Checkpoint Loaded')
            model = load_model(model_name)
        price_train=total_price_train[start:end]
        X_train=total_X_train[start:end]
        y_train=total_y_train[start:end]
        model.fit([X_train,price_train], y_train, epochs=1, batch_size=1)
        model.save(model_name)
        start=end+1
        end=start+siz
    # Final evaluation of the model
    #model = load_model(model_name)
    # print(X_test)
    # print(y_test)
    scores = model.evaluate([X_test,price_test], y_test, verbose=1)
    fd = open('accuracy.csv','a')
    CsvRow = [filename, day, "Accuracy: %.2f%%" % (scores[1]*100)]
    print(CsvRow)
    fd.write(", ".join(CsvRow) + "\n")
    fd.close()

#for f in os.listdir("../data/all_data"):
for f in os.listdir("./"):
    if(f.endswith('.csv')):
        print(f)
        #train(os.path.join("../data/all_data", f), f.replace(".csv","_" + d +"_.hdf5"), d)
        train(os.path.join("./", f), f.replace(".csv","_.hdf5"))
