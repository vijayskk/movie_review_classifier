from sklearn import datasets
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential

data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=100000)
word_index = data.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key, value) in word_index.items() ])

train_data = pad_sequences(train_data,value=word_index["<PAD>"], padding="post",maxlen = 250)
test_data =  pad_sequences(test_data,value=word_index["<PAD>"], padding="post",maxlen = 250)

def decode(text):
    return " ".join([reverse_word_index.get(i , "?") for i in text])

# print(decode(test_data[0]))

# model = Sequential()
# model.add(keras.layers.Embedding(100000,16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(10 , activation='relu'))
# model.add(keras.layers.Dense(1 , activation='sigmoid'))

# model.summary()

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# model.fit(train_data, train_labels, epochs=40, batch_size=512, validation_split=0.2)

# model.save("model.h5")

model = keras.models.load_model("model.h5")

acc = model.evaluate(test_data, test_labels)
print(acc)

# test_rev = test_data[0]
# predict = model.predict([test_rev.reshape(1,250)])
# print("Review: " + decode(test_rev))
# print("Prediction: " + str(predict[0]))
# print("Actual: " + str(test_labels[0]))
def review_encode(s):
    encode = [1]
    for word in s:
        if word.lower() in word_index:
            encode.append(word_index[word.lower()])
        else:
            encode.append(2)    
    return encode

with open("text.txt",encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").replace("?","").replace("!","").replace("-"," ").strip().split(" ")
        encode = review_encode(nline)
        encode = pad_sequences([encode],value=word_index["<PAD>"], padding="post",maxlen = 250)
        pred = model.predict(encode)
        print(line)
        print(pred[0][0])
        if pred[0][0] > 0.5:
            print("Positive")
        else:
            print("Negative")