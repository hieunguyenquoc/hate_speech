import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from keras import backend as K
from keras.models import model_from_json
from keras.utils import pad_sequences


EMBEDDING_FILE = 'Word Embedding/cc.vi.300.vec'
train_x = pd.read_csv('Data/Train.csv').fillna(" ")
# test_x = pd.read_csv('Data/Test.csv').fillna(" ")
# print(type(test_x))
# print(len(train_x))

test_x = ["đánh chết cha mày giờ".lower()]

max_features=7000
maxlen=150
embed_size=300

train_x['free_text'].fillna(' ')

# test_x['free_text'].fillna(' ')
train_y = train_x[['CLEAN', 'OFFENSIVE', 'HATE']].values

train_x = train_x['free_text'].str.lower()

# test_x = test_x['free_text'].str.lower()

# Vectorize text + Prepare  Embedding
tokenizer = text.Tokenizer(num_words=max_features, lower=True)
tokenizer.fit_on_texts(list(train_x))
train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)
# train_x = pad_sequences(train_x, maxlen=maxlen)
# print(len(train_x))
test_x = pad_sequences(test_x, maxlen=maxlen)

print("create vector")
embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue

    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

def get_key(val,dict):
    for key, value in dict.items():
        if val == value:
            return key
        
def create_model():
    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.35)(x)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    out = Dense(3, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Prediction
    # batch_size = 32
    # epochs = 3
    
    return model

model = create_model()
model.load_weights("model.h5")
## fastapi

from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/check/")
async def check(text: str):
    test_x = [text.lower()]
    test_x = tokenizer.texts_to_sequences(test_x)
    test_x = pad_sequences(test_x, maxlen=maxlen)
    predictions = model.predict([test_x], batch_size=1, verbose=1)
    print(predictions)
    dict = {}
    for index, item in enumerate((predictions[0])):
        dict.update({index :item})
    print(dict)
    list_pred = (predictions[0])
    max_element = max(list_pred)
    print(max_element)
    print(type(max_element))
    label = get_key(max_element,dict)
    result = "Unknown"
    if label == 0 :
        result = "CLEAN"
    if label == 1 :
        result = "OFFENSIVE"
    if label == 2:
        result = "HATE"
        
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
