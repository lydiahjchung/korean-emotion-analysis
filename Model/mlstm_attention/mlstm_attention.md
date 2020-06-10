## 감정 데이터셋
모델 학습을 위해 **6가지 카테고리의 감정 데이터**가 라벨링된 데이터셋을 사용하려고 합니다.<br>
6가지 감정 카테고리는 **anger, happiness, neutral, surpirse, sadness, fear, dissgust**로 이루어져 있습니다.
아래 코드는 **google colab**에서 작성되었으며 repository에 업로드 되어있는 python 파일은 아래와 다를 수 있음을 밝힙니다.

```
    import numpy as np 
    import pandas as pd 
    import os.path

    path = os.getcwd()
    col_name = ['text', 'emotions']

    # 데이터 셋 불러오기
    dataset = pd.read_csv(path + "/labeled_final.txt", names=col_name, sep=';')

    dataset
    # 라벨 데이터 히스토그램
    dataset.emotions.value_counts().plot.bar(align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan', "purple"])
```
<div align="center">
  <img src="https://user-images.githubusercontent.com/38775259/84282583-e54ff500-ab74-11ea-9fdb-87b7a3d29583.png" width="400", height="300"></img>
</div>

## 데이터 전처리
데이터 전처리는 용도에 맞게 데이터를 사전에 변경하는 작업입니다. 
**토큰화**, **정규화**, **불용어 처리** 등 여러가지 자연어 처리 기법을 사용하여
데이터 전처리를 진행하려고 합니다.

### 1. 데이터 정제 및 정규화
라벨링된 데이터는 트위터 데이터를 포함하고 있기 때문에 용도에 맞게 데이터를 **정제** 및 **정규화** 해야 합니다. 트위터 아이디 및 링크는 분석 용도에 적합하지 않기 때문에 **정제** 작업을 통해 제거해야 하며, 영어의 표현 방식을 통일하기 위해 데이터 **정규화** 과정을 거쳐야합니다.<br>
- **정제(Cleaning)**: 노이즈 데이터 제거
- **정규화(Normalization)**: 표현 형식 통일
```
    import re

    # 데이터 정규화 및 정제를 위한 함수
    def decontract(phrase):

    # 데이터 정제(Cleaning)
    phrase = re.sub(r"http\S+", "", phrase)                           # 하이퍼링크 제거
    phrase = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', phrase)  # 트위터링크 제거
    phrase = re.sub('@[^\s]+', '', phrase)                           # 트위터아이디 제거

    # 데이터 정규화(Normalization)
    phrase = re.sub('#([^\s]+)', '', phrase)
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", "can not", phrase)
    phrase = re.sub(r"n't", " not", phrase)
    phrase = re.sub(r"'re", " are", phrase)
    phrase = re.sub(r"'s", " is", phrase)
    phrase = re.sub(r"'d", " would", phrase)
    phrase = re.sub(r"'ll", " will", phrase)
    phrase = re.sub(r"'t", " not", phrase)
    phrase = re.sub(r"'ve", " have", phrase)
    phrase = re.sub(r"'m", " am", phrase)
    phrase = re.sub(r"w/", "with", phrase)
    phrase = phrase.replace('.', '')              # 데이터 공백 제거
    phrase = phrase.lower()                       # 데이터 대소문자 통합
    phrase = phrase + "\n"

    return phrase
```

### 2. 토큰화, 어간 추출 및 불용어 처리
이번에는 데이터 전처리 과정에 꼭 필요한 **어간 추출(Stemming)** 및 **불용어 처리(Stop Word)**를 진행할 것입니다.
먼저 **어간 추출**이란 어간(Stem)을 추출하는 작업입니다. 단어는 어간과 어미로 이루어져 있으며 두 요소 중에서 어간을 추출하는 이유는 어간에 단어의 의미가 내포되어 있기 때문입니다.
<br>
다음으로 **불용어 처리**는 문장에서 유의미한 단어를 선택하고 필요 없는 것들은 제거하는 작업을 의미합니다. 예를 들어 문장에서 I, my, me는 자주 등장하는 단어이지만 문장에 자연어 처리 분석에 큰 도움이 되지 않습니다. 이렇게 자주 등장하지만 분석에 도움이 되지 않는 단어들을 제거하는 작업을 불용어 처리라고 합니다.
<br>
해당 프로젝트에서는 nltk 패키지를 사용하여 **토큰화**, **불용어 처리**, **어간 추출**을 진행하였다.
```
    import nltk
    from nltk.stem.snowball import SnowballStemmer
    from string import punctuation 
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt')
    nltk.download('stopwords')

    english_stemmer = SnowballStemmer('english')
    temps = ['``', "''", '…', '—', '~~', '"', '..', '“', '-_____-', 'm̶̲̅ε̲̣', 'rt', '=/', '»']
    itos = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    _stopwords = set(stopwords.words('english') + list(punctuation) + list(range(11)) + temps + itos)

    def pre_to_tok(text):
        text = decontract(text)     # 데이터 정제 및 정규화
        text = word_tokenize(text)  # 데이터 토큰화
        return [english_stemmer.stem(word) for word in text if word not in _stopwords]  # 어간 추출 및 불용어 처리
```
```
    twitter = []        # 트위터 데이터 초기화
    label = []          # 라벨링 데이터 초기화

    # 데이터 불러오기
    with open(path + '/labeled_final.txt', 'r') as f:
    lines = f.readlines()

    # 트위터/라벨링 데이터 리스트로 저장하기
    for line in lines:
        twitter.append(''.join(line.split(';')[0:-1]).replace('\n', ''))  # 트위터 데이터
        label.append(line.split(';')[-1].replace('\n', ''))               # 라벨링 데이터

    processed = []
    # 데이터 전처리
    for sent, em in zip(twitter, label):
        done = pre_to_tok(sent)
        if len(done) != 0:
            processed.append([done, em])
```

### 3. Vocabulary 사전 구성
정수 인코딩을 사용하기 위해 토큰화된 단어를 가지고 Vocabulary를 구성한다. 여기서 Vocabulary란 단수, 복수와 같은 형태는 다르지만 의미는 같은 단어를 같은 단어로 묶어 주는 기법을 의미합니다. 예를 들어 computers는 computer와 같은 단어로 간주합니다. 또한 단어의 빈도 순서대로 정렬하여 인코딩을 하기 위하여 nltk의 FrqDist 클래스를 사용하였습니다.
```
    # Vocabulary 생성 함수
    def build_vocab(data):
        all_words = []

        for words, emotion in data:
            all_words.extend(words)

        wordlist = nltk.FreqDist(all_words)
        word_features = wordlist.keys()
        return wordlist, word_features
    ```
    ```
    # Vocabulary 생성
    wordlist, word_features = build_vocab(processed)

    # 빈도수에 따라 Vocabulary 정렬
    sorted_wordlist = {k: v for k, v in sorted(wordlist.items(), key=lambda item: item[1], reverse=True)}
    sorted_keys = list(sorted_wordlist.keys())
    sorted_keys.append('<unk>')
```

### 4. 정수 인코딩
컴퓨터는 기본적으로 텍스트 데이터 보다 숫자 데이터를 훨씬 더 빠르게 처리할 수 있습니다. 따라서 자연어 처리에서는 앞서 만든 Vocabulary를 정수로 변환하는 기법을 사용합니다. 인덱스를 부여하는 방법에는 여러 가지 방법이 있지만 앞서 Vocabulary를 빈도수로 정렬하였기 때문에 빈도수를 기준으로 인덱싱을 진행하였습니다.
```
    # 감정 라벨
    emotion = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'neutral', 'disgust']

    # 정수 인코딩
    vectorized = []
    for sent, emo in processed:
        vect = []
        for word in sent:
            vect.append(sorted_keys.index(word)+1)
        vectorized.append([vect, emotion.index(emo)])
```
### 5. 데이터 분리
test 데이터 셋과 train 데이터 셋을 위하여 데이터 분리를 진행했습니다. 현재 데이터가 빈도 수에 따라 정렬되어 있기 때문에 random.shuffle을 통해 순서를 바꿔준 후 데이터를 분리하였습니다.
```
    import random
    import keras
    import numpy as np
    from collections import Counter

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
```

## mLSTM + Attention 모델 정의
이제 본격적으로 모델을 구현하는 파트입니다. 해당 프로젝트 에서는 keras를 사용하여 모델을 구현하였으며 세부적인 특징은 다음과 같습니다.
- **Loss Function**: Sparse categorical crossentropy<br>
Sparse categorical crossentropy의 경우 다중 분류 손실함수로, categorical crossentropy와는 다르게 one-hot 인코딩을 할 필요가 없습니다. 현재 정수 타입 인코딩을 진행하였기 때문에 sparse categorical crossentropy를 손실함수로 사용하였습니다.
- **Optimizer**: Adam
Adam optimizer는 stepsize가 gradient의 rescaling에 영향을 받지 않는 장점을 가지고 있습니다. 즉, gradient가 커져도 bound가 되어 있어서 어떠한 objective function을 사용한다 하더라도 안정적으로 최적화를 할 수 있습니다. 따라서 optimizer로 Adam method를 사용하였습니다.
```
    import keras

    embedding_dim = 100 # The dimension of word embeddings

    # Define input tensor
    sequence_input = keras.Input(shape=(max_words,), dtype='int32')

    # Word embedding layer
    embedded_inputs =keras.layers.Embedding(len(sorted_keys) + 1,
                                            embedding_dim,
                                            input_length=max_words)(sequence_input)

    # Apply dropout to prevent overfitting
    embedded_inputs = keras.layers.Dropout(0.1)(embedded_inputs)

    # Apply Bidirectional mLSTM over embedded inputs
    lstm_outs = keras.layers.wrappers.Bidirectional(
        MultiplicativeLSTM(embedding_dim, return_sequences=True)
    )(embedded_inputs)
    
    # Apply Bidirectional LSTM over embedded inputs
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
```
<div align="center">
  <img src="https://user-images.githubusercontent.com/38775259/84282858-3fe95100-ab75-11ea-8106-120f158d44a6.png" width="ㅕ800", height="500"></img>
</div>

## 모델 훈련
앞서 정의한 모델에 test/train 데이터를 활용하여 모델을 훈련시켰습니다. Overfitting을 방지하기 위하여 keras의 EarlyStopping을 사용하였습니다. 모델이 이미 존재하는 경우 훈련된 모델을 불러왔으며, 새롭게 training 하는 경우 test/validiaton set의 loss/accuracy를 시각화하였다.
```
    import os.path
    from keras.models import load_model
    from keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt

    # 현재 경로
    path = os.getcwd() + "/lstm_attention_v1_trained.h5"

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
```
<div align="center">
  <img src="https://user-images.githubusercontent.com/38775259/84282952-5abbc580-ab75-11ea-86b6-bb26a1cf9755.png" width="400", height="300"></img>
</div>

## 모델 테스트
실제 test 데이터를 사용하여 모델을 test하였습니다. 크롤링한 트위터 데이터 100개와, 영화 평론 데이터 100개의 총 200개의 데이터를 test 데이터로 사용하였으며 각각의 결과를 csv로 저장하였습니다.
```
    def test_preprocess(platform):
        with open("{}_final_test.txt".format(platform)) as f:
            testdata = f.readlines()

        final_test, processed_test = [], []

        # 데이터 한 줄씩 읽기
        for line in testdata:
            final_test.append(line.strip())

        # 데이터 전처리
        for each in final_test:
            done = pre_to_tok(each)
            if len(done) != 0:
                processed_test.append(done)

        # 데이터 토큰화 및 패딩
        test_vect = []
        for sent in processed_test:
            vect = []
            for word in sent:
            if word in sorted_keys:
                vect.append(sorted_keys.index(word)+1)
            else:
                vect.append(len(sorted_keys))
            test_vect.append(vect)
        test_vect = keras.preprocessing.sequence.pad_sequences(test_vect, maxlen=max_words)
        return test_vect, final_test

    # NMT API 선택
    platform = "kakao"
    vector_result, final_result = test_preprocess(platform)
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

    pred
```
```
    # CSV 저장
    pred.to_csv("{}_predicted.csv".format(platform),
                    sep=';',
                    columns = ['sentence', 'label', 'prob'],
                    index = False)
```

## Attention 결과 출력
Attention mechanism을 적용 결과를 시각화하기 위해 matplotlib을 사용하였습니다. Test 데이터에서 랜덤하게 문장 하나를 선택하여 결과값을 출력하였습니다.
```
    # Re-create the model to get attention vectors as well as label prediction
    model_with_attentions = keras.Model(inputs=model.input,
                                        outputs=[model.output, 
                                                model.get_layer('attention_vec').output])
```
```
    emotion = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'neutral', 'disgust']
    label2id = dict(zip(emotion, [0, 1, 2, 3, 4, 5, 6]))
    id2label = dict(zip([0, 1, 2, 3, 4, 5, 6], emotion))
    sample_text = pre_to_tok(random.choice(dataset["text"].values))
    print(sample_text)
```
```
    import random
    import math

    #Select random samples to illustrate
    #sentence = random.choice(dataset["text"].values)
    sentence = random.choice(final_result)
    tokenized_sample = pre_to_tok(sentence)

    encoded_samples = []
    for word in tokenized_sample:
    pre_word = pre_to_tok(word)
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
    import matplotlib.pyplot as plt; plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.core.display import display, HTML

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
```
<div align="center">
  <img src="https://user-images.githubusercontent.com/38775259/84283188-ab332300-ab75-11ea-8eba-76d24e3de1ec.png" width="800", height="300"></img>
</div>
