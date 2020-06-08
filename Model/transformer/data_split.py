from tensorflow import keras
import numpy as np
import random
from preprocess import Preprocess

class Splitting:
    def __init__(self):
        self.labels = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'neutral', 'disgust']
        self.each = {'happiness': [],
                     'sadness': [],
                     'anger': [],
                     'fear': [],
                     'surprise': [],
                     'neutral': [],
                     'disgust': []}
        self.max_len = 35
        self.trains, self.tests = [], []
        self.pre = Preprocess()

    def each_label(self, total):
        ''' splitting entire data by each label
        returns preprocessed data by each label '''
        for sentence, emotion in total:
            if emotion == self.labels[0]:
                self.each[emotion].append([sentence, self.labels.index(emotion)])
            elif emotion == self.labels[1]:
                self.each[emotion].append([sentence, self.labels.index(emotion)])
            elif emotion == self.labels[2]:
                self.each[emotion].append([sentence, self.labels.index(emotion)])
            elif emotion == self.labels[3]:
                self.each[emotion].append([sentence, self.labels.index(emotion)])
            elif emotion == self.labels[4]:
                self.each[emotion].append([sentence, self.labels.index(emotion)])
            elif emotion == self.labels[5]:
                self.each[emotion].append([sentence, self.labels.index(emotion)])
            else:
                self.each[emotion].append([sentence, self.labels.index(emotion)])

    def shuffle(self):
        for i in range(len(self.labels)):
            label = self.labels[i]
            each_label = self.each[label]
            split_at = round(len(each_label) * 0.7)
            random.shuffle(each_label)
            self.trains.extend(each_label[:split_at])
            self.tests.extend(each_label[split_at:])

    def train_and_test(self):
        x_train, y_train, x_val, y_val = [], [], [], []

        for x, y in self.trains:
            x_train.append(x)
            y_train.append(y)

        for x, y in self.tests:
            x_val.append(x)
            y_val.append(y)

        return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)

    def padding(self, data):
        return keras.preprocessing.sequence.pad_sequences(data, maxlen=self.max_len)

    def predictions(self, original, predicted):
        sentences, labels, probabilities = [], [], []

        for each in original:
            sentences.append(each.strip())

        tmp_label = list(predicted.argmax(axis=-1))

        for label in tmp_label:
            labels.append(self.labels[label])

        for probability in predicted:
            probabilities.append(max(probability))

        return sentences, labels, probabilities