import json
import string
import random
import pandas as pd
import numpy as np
import tensorflowjs as tfjs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Input, Flatten
from keras.models import Model
from sklearn.preprocessing import LabelEncoder

with open('/home/jose2808/ejemploClase10/modelo_python/intents.json') as content:
    data = json.load(content)

tags = []
patterns = []
responses = {}

for intent in data['intents']:
    responses[intent['tag']] = intent['responses']

    for line in intent['patterns']:
        patterns.append(line)
        tags.append(intent['tag'])

data = pd.DataFrame({"patterns": patterns, "tags":tags})
data['patterns'] = data['patterns'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])

x_train = pad_sequences(train)

le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]

vocabulary = len(tokenizer.word_index)
output_lenght = le.classes_.shape[0]

i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1, 10)(i)
x = LSTM(10,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_lenght,activation='softmax')(x)
model = Model(i, x)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200)

print(model.summary())

tfjs.converters.save_keras_model(model, './modelo_python')

with open('./modelo_python/tokenizer.json', 'w') as f:
    json.dump(json.loads(tokenizer.to_json()), f)

# while True:
#     texts_p = []
#     prediction_input = input('You : ')

#     prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
#     prediction_input = ''.join(prediction_input)
#     texts_p.append(prediction_input)

#     prediction_input = tokenizer.texts_to_sequences(texts_p)
#     prediction_input = np.array(prediction_input).reshape(-1)
#     prediction_input = pad_sequences([prediction_input], input_shape)

#     output = model.predict(prediction_input)
#     output = output.argmax()

#     response_tag = le.inverse_transform([output])[0]
#     print("Chatbot : ", random.choice(responses[response_tag]))
