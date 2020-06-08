# Multi-class emotion classification using Transformer

## Using Repository Files

- preprocess.py
- layers.py
- data_split.py
- transformer.py

To simply run the model, run **transformer.py** file. The remaining .py files are for each classes used in each steps of building the model:
1. Text Preprocessing
2. Layer Implementation
3. Splitting Data

For specified details of each steps, read the following section: ***Using Google Colab***.

## Using Google Colab

Detailed order of the implementation process is delineated below.
This code is limitting its output to *Kakao NMT translated data set*. For other outputs using Google or Papago NMT, alter **{NMT API NAME}** in section **Predicting label classification of the test data set**.

### Data Preprocessing

The labeled data used for this emotion analysis has **seven different labels**: *anger; happiness; neutral; sadness; surprise; fear; disgust*.

#### Setup
```python
    import re
    import nltk
    from nltk.tokenize import word_tokenize
    from string import punctuation
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download('punkt')
    nltk.download('stopwords')
```
#### Loading the data
```python
    # loading the original labeled data
    with open("labeled_final.txt") as f:
        data = f.readlines()

    # labeled = [[sentence, emotion], [sentence, emotion], ... ]
    labeled = []
    for line in data:
        cut_idx = len(line) - line[::-1].find(";")
        sent, em = line[:cut_idx-1], line[cut_idx:].strip()
        labeled.append([sent, em])
```
#### Cleaning and Normalizing

For twitter text datas in the training set, regular expression is used to remove usernames, hashtags, and URLs. Abbreviations are also decontracted.

```python
    def decontracting(phrase):
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"n\'t", " not", phrase)
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
        return phrase

    def cleaning(text):
        ''' regex, decontraction, tokenizing, stemming in one go '''
        text = text.lower()
        text = re.sub('@[^\s]+', '', text)
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
        text = re.sub('#([^\s]+)', '', text)
        text = decontracting(text)
        return text
```
#### Removing stopwords, Stemming, and Tokenizing

Using nltk's stopwords and punctuations, and Porter Stemmer, each sentences was stemmed and tokenized after removing unnecessary stopwords.

```python
    porter = PorterStemmer()

    tmps = ['``', "''", '...', '--', '~~', '"', '..', '“', '-_____-', 'm̶̲̅ε̲̣', 'rt', '=/', '»']
    itos = ['0','1','2','3','4','5','6','7','8','9','10']

    STOPWORDS = set(stopwords.words('english') + list(punctuation) + list(range(11)) + tmps + itos)

    def preprocessing(text):
        text = word_tokenize(text)
        return [porter.stem(word) for word in text if word not in STOPWORDS]
```
```python  
    processed = []

    for sentence, emotion in cleaned:
        done = preprocessing(sentence)
        if len(done) != 0:
            processed.append([done, emotion])
```
#### Building Vocabulary

The frequency of each stemmed words was counted to build a vocabulary. This vocabulary is further used when each words is integer encoded.

```python
    def build_vocab(data):
        all_words = []

        for words, emotion in data:
          all_words.extend(words)

        wordlist = nltk.FreqDist(all_words)
        word_features = wordlist.keys()
        return wordlist, word_features

    wordlist, word_features = build_vocab(processed)
    sorted_wordlist = {k: v for k, v in sorted(wordlist.items(), key=lambda item: item[1], reverse=True)}
    vocab = list(sorted_wordlist.keys())
    vocab.append('<UNK>')
```
#### Splitting data set and Integer Encoding

In order to keep the proportion of each label data in train and validation set, the data was first separated by each labels. Train set and validation set were split in 7:3 ratio. Each integer encoded sentences was then padded to fit the maximum length of the sentences.

```python
# labels
EMOTION = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'neutral', 'disgust']

# maximum length of each sentences
max_len = 35

def vectorize(sentence):
  vect = []
  for word in sentence:
    vect.append(vocab.index(word) + 1)
  return vect

def splitting(train, val, label):
  idx = round(len(label) * 0.7)
  train.extend(label[:idx])
  val.extend(label[idx:])
```
```python
# splitting the data by label and
# integer encoding the words based on the vocab
happiness, sadness, anger, fear, surprise, neutral, disgust = [], [], [], [], [], [], []
for sentence, emotion in processed:
  if emotion == 'happiness':
    happiness.append([vectorize(sentence), 0])
  elif emotion == 'sadness':
    sadness.append([vectorize(sentence), 1])
  elif emotion == 'anger':
    anger.append([vectorize(sentence), 2])
  elif emotion == 'fear':
    fear.append([vectorize(sentence), 3])
  elif emotion == 'surprise':
    surprise.append([vectorize(sentence), 4])
  elif emotion == 'neutral':
    neutral.append([vectorize(sentence), 5])
  else:
    disgust.append([vectorize(sentence), 6])
```
```python
# shuffle each label data and
# equally split each label into train and validation test set
train, val = [], []

random.shuffle(happiness)
splitting(train, val, happiness)

random.shuffle(sadness)
splitting(train, val, sadness)

random.shuffle(anger)
splitting(train, val, anger)

random.shuffle(fear)
splitting(train, val, fear)

random.shuffle(surprise)
splitting(train, val, surprise)

random.shuffle(neutral)
splitting(train, val, neutral)

random.shuffle(disgust)
splitting(train, val, disgust)

random.shuffle(train)
random.shuffle(val)
```
```python
# splitting x, y in train and val
x_train, y_train, x_val, y_val = [], [], [], []

for x, y in train:
  x_train.append(x)
  y_train.append(y)

for x, y in val:
  x_val.append(x)
  y_val.append(y)

x_train, y_train, x_val, y_val = np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)

# padding x with max_len
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=max_len)
```

### Implementing Layers

#### Multi Head Self Attention Layer
```python
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
```

#### Transformer Block Layer
```python
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
```

#### Embedding Layer
```python
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
```

### Training Transformer

#### Classifier model using transformer layer
```python
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = keras.Input(shape=(max_len,),)
embedding_layer = TokenAndPositionEmbedding(max_len, len(vocab)+1, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(7, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

```python
model.summary()
```

#### Training and Evaluating
```python
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping()
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), callbacks=[early_stopping]
)
```

### Predicting label classification of the test data set

#### Preprocessing test data
```python
# loading {KAKAO} test data
with open("kakao_final_test.txt") as f:
  kakao_input = f.readlines()

# text preprocessing the test data
kakao_original, kakao_processed = [], []

for line in kakao_input:
  kakao_original.append(line.strip())

for each in kakao_original:
  done = preprocessing(cleaning(each))
  if len(done) != 0:
      kakao_processed.append(done)

# integer encoding the test data based on the trained vocab
kakao_encoded = []

for sentence in kakao_processed:
  vect = []
  for word in sentence:
    if word in vocab:
      vect.append(vocab.index(word) + 1)
    else:
      vect.append(len(vocab))
  kakao_encoded.append(vect)

# padding each sentences in test data
kakao_processed = keras.preprocessing.sequence.pad_sequences(kakao_encoded, maxlen=max_len)
```

#### Predicting labels and Saving output
```python
# predicting the labels for {KAKAO} test data
kakao_predict = model.predict(kakao_processed)
```

```python
# saving predicted label and probability of each sentences
kakao_labels, kakao_probs = [], []

# saving predicted label
tmp_labels = list(kakao_predict.argmax(axis=-1))

for idx in tmp_labels:
  kakao_labels.append(EMOTION[idx])

# saving predicted probability
for each in kakao_predict:
  kakao_probs.append(max(each))
```

```python
# saving output results as csv file
import pandas as pd

kakao_transformer = pd.DataFrame(
    {'sentence': kakao_original,
     'label': kakao_labels,
     'probability' : kakao_probs
    })

kakao_transformer.to_csv("kakao_transformer.csv",
                  sep=';',
                  columns = ['sentence', 'label', 'probability'],
                  index = False)
```
