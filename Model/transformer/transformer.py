import re, nltk, random
import numpy as np
from string import punctuation 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from keras.callbacks import EarlyStopping

nltk.download('punkt')
nltk.download('stopwords')

english_stemmer = SnowballStemmer('english')
EMOTION = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'neutral', 'disgust']
max_len = 35

class TF_Preprocess:
    def __init__(self):
        temps = ['``', "''", '...', '--', '~~', '"', '..', '“', '-_____-', 'm̶̲̅ε̲̣', 'rt', '=/', '»']
        itos = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self._stopwords = set(stopwords.words('english') + list(punctuation) + list(range(11)) + temps + itos)

    def decontract(self, phrase):
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

    def pre_to_tok(self, text):
        '''Data preprocessing; replacements, decontraction, stemming'''
        text = text.lower()
        text = re.sub('@[^\s]+', '', text)
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
        text = re.sub('#([^\s]+)', '', text)
        text = self.decontract(text)
        text = word_tokenize(text)
        return [english_stemmer.stem(word) for word in text if word not in self._stopwords]

# Saving labeled_final.txt data
with open("labeled_final.txt") as f:
    txt_data = f.readlines()

# Splitting text data into lists
labeled = []
for line in txt_data:
    cut_idx = len(line) - line[::-1].find(";")
    sent, em = line[:cut_idx-1], line[cut_idx:].strip()
    labeled.append([sent, em])

# Preprocessing labaled list
processed, TFP = [], TF_Preprocess()
for sent, em in labeled:
    done = TFP.pre_to_tok(sent)
    if len(done) != 0:
        processed.append([done, em])

# Building vocabulary based on text
def build_vocab(data):
    all_words = []

    for words, emotion in data:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    return wordlist, word_features

wordlist, word_features = build_vocab(processed)
sorted_wordlist = {k: v for k, v in sorted(wordlist.items(), key=lambda item: item[1], reverse=True)}
sorted_keys = list(sorted_wordlist.keys())

# Map words to index values based on the vocabulary
vectorized = []
for sent, em in processed:
    vect = []
    for word in sent:
        vect.append(sorted_keys.index(word)+1)
    vectorized.append([vect, EMOTION.index(em)])

# Split data into train set and validation set by 0.75:0.25
random.shuffle(vectorized)
trains, vals = vectorized[:10422], vectorized[10422:]

x_train, y_train, x_val, y_val = [], [], [], []

for x, y in trains:
  x_train.append(x)
  y_train.append(y)

for x, y in vals:
  x_val.append(x)
  y_val.append(y)

x_train, y_train, x_val, y_val = np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)

######################### M O D E L I N G ######################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiheadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiheadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension should be divisible by number of heads")
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
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiheadSelfAttention(embed_dim, num_heads)
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


vocab_size = len(sorted_keys)
maxlen = 35
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

embed_dim = 32 
num_heads = 2 
ff_dim = 32 

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(7, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

early_stopping = EarlyStopping()
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), callbacks=[early_stopping]
)