import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Embedding, LSTM, Input
from keras.models import Model

## DATA
lines = open('/home/jose2808/ejemploClase4/corpus/movie_lines.txt', 
             encoding='UTF-8', errors='ignore').read().split('\n')
convers = open('/home/jose2808/ejemploClase4/corpus/movie_conversations.txt', 
               encoding='UTF-8', errors='ignore').read().split('\n')

exchn = []
for conver in convers:
    exchn.append(conver.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(",", "").split())

diag = {}
for line in lines:
    line_splitted = line.split(' +++$+++ ')
    diag[line_splitted[0]] = line_splitted[-1]

questions = []
answers = []

for conver in exchn:
    for i in range(len(conver) - 1):
        questions.append(diag[conver[i]])
        answers.append(diag[conver[i + 1]])

del(conver, convers, diag, exchn, i, line, lines)

sorted_ques = []
sorted_ans = []
for i in range(len(questions)):
    if len(questions[i]) < 13:
        sorted_ques.append(questions[i])
        sorted_ans.append(answers[i])


def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt

clean_ques = []
clean_ans = []

for line in sorted_ques:
    clean_ques.append(clean_text(line))

for line in sorted_ans:
    clean_ans.append(clean_text(line))

for i in range(len(clean_ans)):
    clean_ans[i] = ' '.join(clean_ans[i].split()[:11])

del(sorted_ans, sorted_ques)

clean_ans = clean_ans[:1500]
clean_ques = clean_ques[:1500]

word2count = {}

for line in clean_ques:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for line in clean_ans:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


del(word, line)

thresh = 5
vocab = {}
word_num = 0
for word, count in word2count.items():
    if count >= thresh:
        vocab[word] = word_num
        word_num += 1


del(word2count, word, count, thresh)       
del(word_num) 

for i in range(len(clean_ans)):
    clean_ans[i] = '<SOS> ' + clean_ans[i] + ' <EOS>'

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
#ERROR: En lugar de ser len(tockens) era len(vocab)
x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1

vocab['cameron'] = vocab['<PAD>']
vocab['<PAD>'] = 0

del(token, tokens) 
del(x)

inv_vocab = {w:v for v, w in vocab.items()}

del(i)

encoder_inp = []
for line in clean_ques:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
        
    encoder_inp.append(lst)

decoder_inp = []
for line in clean_ans:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
    decoder_inp.append(lst)

del(clean_ans, clean_ques, line, lst, word)

encoder_inp = pad_sequences(encoder_inp, 13, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, 13, padding='post', truncating='post')

decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:])

decoder_final_output = pad_sequences(decoder_final_output, 13, padding='post', truncating='post')

del(i)

decoder_final_output = to_categorical(decoder_final_output, len(vocab))

### ------------------------------------------------------- MODELO 1 -----------------------------------------------------------
enc_inp = Input(shape=(13,))
dec_inp = Input(shape=(13,))

VOCAB_SIZE = len(vocab)
embed = Embedding(VOCAB_SIZE+1, output_dim=50, input_length=13, trainable=True)

enc_embed = embed(enc_inp)
enc_lstm = LSTM(400, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h, c]

dec_embed = embed(dec_inp)
dec_lstm = LSTM(400, return_sequences=True, return_state=True)
#Error en esta línea
#dec_op, _, _ = dec_lstm(enc_embed)
dec_op, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

dense = Dense(VOCAB_SIZE, activation='softmax')

dense_op = dense(dec_op)

model = Model([enc_inp, dec_inp], dense_op)

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

model.fit([encoder_inp, decoder_inp], decoder_final_output, epochs=10)

### ------------------------------------------------------- MODELO 2 -----------------------------------------------------------
enc_model = Model([enc_inp], enc_states)

decoder_state_input_h = Input(shape=(400,))
decoder_state_input_c = Input(shape=(400,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


decoder_outputs, state_h, state_c = dec_lstm(dec_embed, 
                                    initial_state=decoder_states_inputs)


decoder_states = [state_h, state_c]

#Error: la variable con la que se construía el modelo era decoder_states, no decoder_outputs
#dec_model = Model([dec_inp] + decoder_states_inputs, [decoder_outputs] + decoder_outputs)
dec_model = Model([dec_inp] + decoder_states_inputs, [decoder_outputs] + decoder_states)

print("##########################################")
print("#       start chatting ver. 1.0          #")
print("##########################################")

prepro1 = ""
while prepro1 != 'q':
    prepro1 = input("you : ")
    prepro1 = clean_text(prepro1)
    prepro = [prepro1]
    
    txt = []
    for x in prepro:
        lst = []
        for y in x.split():
            try:
                lst.append(vocab[y])
            except:
                lst.append(vocab['<OUT>'])
        txt.append(lst)

    #A partir de esta línea, el código ya no se encuentra dentro del ciclo for
    txt = pad_sequences(txt, 13, padding='post')
    stat = enc_model.predict( txt )

    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = vocab['<SOS>']

    stop_condition = False
    decoded_translation = ''

    while not stop_condition :

        dec_outputs, h, c = dec_model.predict([empty_target_seq] + stat)
        decoder_concat_input = dense(dec_outputs)

        sample_word_index = np.argmax(decoder_concat_input[0, -1, :])
        #Error, en lugar de inv_vocab[sample_word_index] = ' ' era inv_vocab[sample_word_index] + ' '
        #sampled_word = inv_vocab[sample_word_index] = ' '
        sampled_word = inv_vocab[sample_word_index] + ' '

        if sampled_word != '<EOS>':
            decoded_translation += sampled_word

        if sampled_word == '<EOS>' or len(decoded_translation.split()) > 13:
            stop_condition = True

        empty_target_seq = np.zeros((1,1))
        empty_target_seq[0,0] = sample_word_index
        stat=[h,c]

    print("chatbot: ", decoded_translation)


