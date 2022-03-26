from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences


data = keras.datasets.imdb
word_index = data.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


model = keras.models.load_model("model.h5")


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