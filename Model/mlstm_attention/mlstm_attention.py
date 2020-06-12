import numpy as np 
import pandas as pd
import nltk
import random
import keras
import numpy as np
import os.path
import matplotlib.pyplot as plt

from collections import Counter
from keras.models import load_model
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt; plt.rcdefaults()
from IPython.core.display import display, HTML

# 모듈 함수
import function.Constant as current_path
from function.pre_process import PreProcess
from model.multiplicative_lstm import MultiplicativeLSTM

#------------------------------------------------------------------------------------------#
#  1. 감정 데이터 셋
#------------------------------------------------------------------------------------------# 
col_name = ['text', 'emotions']
dataset = pd.read_csv(current_path.data_path + "/labeled_final.txt",       # 데이터 셋 불러오기
                      names=col_name, sep=';')
dataset.emotions.value_counts().plot.bar(align='center', alpha=0.5,   # 라벨 데이터 히스토그램
                                         color=['black', 'red', 'green', 'blue', 'cyan', "purple"])

#------------------------------------------------------------------------------------------#
# 2-2. 토큰화, 어간 추출 및 불용어 처리
#------------------------------------------------------------------------------------------#
twitter = []        # 트위터 데이터 초기화
label = []          # 라벨링 데이터 초기화

# 데이터 불러오기
with open(current_path.data_path + '/labeled_final.txt', 'r', encoding='utf-8') as f:
  lines = f.readlines()

# 트위터/라벨링 데이터 리스트로 저장하기
for line in lines:
  twitter.append(''.join(line.split(';')[0:-1]).replace('\n', ''))  # 트위터 데이터
  label.append(line.split(';')[-1].replace('\n', ''))               # 라벨링 데이터

processed = []
# 데이터 전처리
for sent, em in zip(twitter, label):
    done = PreProcess().pre_to_tok(sent)
    if len(done) != 0:
        processed.append([done, em])

#------------------------------------------------------------------------------------------#
# 2-3. Vocabulary 사전 구성
#------------------------------------------------------------------------------------------#
# Vocabulary 생성
wordlist, word_features = PreProcess().build_vocab(processed)

# 빈도수에 따라 Vocabulary 정렬
sorted_wordlist = {k: v for k, v in sorted(wordlist.items(), key=lambda item: item[1], reverse=True)}
sorted_keys = list(sorted_wordlist.keys())
sorted_keys.append('<unk>')

#------------------------------------------------------------------------------------------#
# 2.4  정수 인코딩
#------------------------------------------------------------------------------------------#
# 감정 라벨
emotion = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'neutral', 'disgust']

# 정수 인코딩
vectorized = []
for sent, emo in processed:
  vect = []
  for word in sent:
    vect.append(sorted_keys.index(word)+1)
  vectorized.append([vect, emotion.index(emo)])

#------------------------------------------------------------------------------------------#
# 2.5 데이터 분리
#------------------------------------------------------------------------------------------#
# 데이터 순서 변경
random.shuffle(vectorized)
trains, tests = vectorized[:11117], vectorized[11117:]
print(len(trains), len(tests))

# 데이터 최대 길이
review_lens = Counter([len(x) for x, y in vectorized])
max_words = max(review_lens)

# Train, Test 데이터 셋 만들기
x_train, y_train, x_val, y_val = [], [], [], []

for x, y in trains:
  x_train.append(x)
  y_train.append(y)

for x, y in tests:
  x_val.append(x)
  y_val.append(y)

# Numpy로 자료형 변형
x_train, y_train, x_val, y_val = np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)

# 데이터셋 패딩
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_words)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=max_words)

#------------------------------------------------------------------------------------------#
# 3. mLSTM + Attention 모델 정의 
#------------------------------------------------------------------------------------------#
embedding_dim = 100 # The dimension of word embeddings

# Define input tensor
sequence_input = keras.Input(shape=(max_words,), dtype='int32')

# Word embedding layer
embedded_inputs =keras.layers.Embedding(len(sorted_keys) + 1,
                                        embedding_dim,
                                        input_length=max_words)(sequence_input)

# Apply dropout to prevent overfitting
embedded_inputs = keras.layers.Dropout(0.1)(embedded_inputs)

# mLSTM 모델
lstm_outs = keras.layers.wrappers.Bidirectional(
    MultiplicativeLSTM(embedding_dim, return_sequences=True)
)(embedded_inputs)

# LSTM 모델
# lstm_outs = keras.layers.wrappers.Bidirectional(
#     keras.layers.LSTM(embedding_dim, return_sequences=True)
# )(embedded_inputs)

# Apply dropout to LSTM outputs to prevent overfitting
lstm_outs = keras.layers.Dropout(0.2)(lstm_outs)

# Attention Mechanism - Generate attention vectors
input_dim = int(lstm_outs.shape[2])
permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
attention_vector = keras.layers.Reshape((max_words,))(attention_vector)
attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

# Last layer: fully connected with softmax activation
fc = keras.layers.Dense(embedding_dim, activation='relu')(attention_output)
output = keras.layers.Dense(len(emotion), activation='softmax')(fc)

# Finally building model
model = keras.Model(inputs=[sequence_input], outputs=output)
model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

# Print model summary
model.summary()

#------------------------------------------------------------------------------------------#
# 4. 모델 훈련
#------------------------------------------------------------------------------------------#
# 현재 경로
path = current_path.data_path + "/lstm_attention_v1_trained.h5"

# pre-trained 데이터가 있는지 확인
if os.path.isfile(path):
  print("Trained model already exists")
  model = load_model('lstm_attention_v1_trained.h5')
else:
  # 모델 학습
  early_stopping = EarlyStopping()             # overfitting 방지
  hist = model.fit(x_train, y_train, epochs=64, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])
  model.save("lstm_attention_v1_trained.h5")   # 모델 저장

  # 모델 evaluation
  loss_and_metrics = model.evaluate(x_val, y_val, batch_size=32)
  print('## evaluation loss and_metrics ##')
  print(loss_and_metrics)

  # 결과 시각화
  fig, loss_ax = plt.subplots()
  acc_ax = loss_ax.twinx()

  loss_ax.plot(hist.history['loss'], 'y', label='train loss')
  loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
  loss_ax.set_xlabel('epoch')
  loss_ax.set_ylabel('loss')
  loss_ax.legend(loc='upper left')

  acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
  acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
  acc_ax.set_ylabel('accuracy')
  acc_ax.legend(loc='upper right')

  plt.show()

#------------------------------------------------------------------------------------------#
# 5. 모델 테스트
#------------------------------------------------------------------------------------------#
# NMT API 선택
platform_list = ["kakao", "google", "papago"]
platform = "kakao"
vector_result, final_result = PreProcess().test_preprocess(platform, sorted_keys, max_words)
preds = model.predict(vector_result)

# label, sentence 데이터 설정
lbl = list(model.predict(vector_result).argmax(axis=-1))
label, probs, i = [], [], 0
for each in lbl:
  label.append(emotion[each])
for each in preds:
  probs.append(max(each))
  i += 1

# 데이터 프레임으로 데이터 출력
import pandas as pd
pred = pd.DataFrame(
    {'sentence': final_result,
     'label': label,
     'prob' : probs
    })

# 데이터 프레임 CSV로 저장
pred.to_csv(current_path.result_path + "/{}_mlstm_attention.csv".format(platform),
                  sep=';',
                  columns = ['sentence', 'label', 'prob'],
                  index = False)
#------------------------------------------------------------------------------------------#
# 7. Attention 결과 출력
#------------------------------------------------------------------------------------------#
# Re-create the model to get attention vectors as well as label prediction
model_with_attentions = keras.Model(inputs=model.input,
                                    outputs=[model.output, 
                                             model.get_layer('attention_vec').output])

emotion = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'neutral', 'disgust']
label2id = dict(zip(emotion, [0, 1, 2, 3, 4, 5, 6]))
id2label = dict(zip([0, 1, 2, 3, 4, 5, 6], emotion))

sample_text = PreProcess().pre_to_tok(random.choice(dataset["text"].values))
print(sample_text)

import random
import math

#Select random samples to illustrate
sentence = random.choice(dataset["text"].values)
tokenized_sample = PreProcess().pre_to_tok(sentence)

encoded_samples = []
for word in tokenized_sample:
  pre_word = PreProcess().pre_to_tok(word)
  if len(pre_word) == 0:
    encoded_samples.append(sorted_keys.index('<unk>')+1)
  elif pre_word[0] not in sorted_keys:
    encoded_samples.append(sorted_keys.index('<unk>')+1)
  else:
    encoded_samples.append(sorted_keys.index(pre_word[0])+1)

encoded_samples = np.array([encoded_samples])
print(encoded_samples)

# Padding
encoded_samples = keras.preprocessing.sequence.pad_sequences(encoded_samples, maxlen=max_words)
print(model.predict(encoded_samples))

# Make predictions
label_probs, attentions = model_with_attentions.predict(encoded_samples)
label_probs = {id2label[_id]: prob for (label, _id), prob in zip(label2id.items(),label_probs[0])}

# Get word attentions using attenion vector
token_attention_dic = {}
max_score = 0.0
min_score = 0.0
for token, attention_score in zip(tokenized_sample, attentions[0][-len(tokenized_sample):]):
    token_attention_dic[token] = math.sqrt(attention_score)


# VISUALIZATION
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
    
def attention2color(attention_score):
    r = 255 - int(attention_score * 255)
    color = rgb_to_hex((255, r, r))
    return str(color)
    
# Build HTML String to viualize attentions
html_text = "<hr><p style='font-size: large'><b>Text:  </b>"
for token, attention in token_attention_dic.items():
    html_text += "<span style='background-color:{};'>{} <span> ".format(attention2color(attention),
                                                                        token)
html_text += "</p>"
# Display text enriched with attention scores 
display(HTML(html_text))
print("Original Sentence: ", sentence)

# PLOT EMOTION SCORES
emotions = [label for label, _ in label_probs.items()]
scores = [score for _, score in label_probs.items()]
plt.figure(figsize=(10,2))
plt.bar(np.arange(len(emotions)), scores, align='center', alpha=0.5, color=['yellow', 'red', 'green', 'blue', 'cyan', "purple", "orange"])
plt.xticks(np.arange(len(emotions)), emotions)
plt.ylabel('Scores')
plt.show()