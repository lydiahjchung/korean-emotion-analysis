import re
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

class Preprocess:
    def __init__(self):
        # nltk.download('punkt')
        # nltk.download('stopwords')

        tmps = ['``', "''", '...', '--', '~~', '"', '..', '“', '-_____-', 'm̶̲̅ε̲̣', 'rt', '=/', '»']
        itos = ['0','1','2','3','4','5','6','7','8','9','10']

        self._stopwords = set(stopwords.words('english') + list(punctuation) + list(range(11)) + tmps + itos)
        self.english_stemmer = PorterStemmer()
        self.word_freqs = dict()
        self.vocab = []

    def decontracting(self, phrase):
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

    def cleaning(self, text):
        ''' regex, decontraction, tokenizing, stemming in one go '''
        text = text.lower()
        text = re.sub('@[^\s]+', '', text)
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
        text = re.sub('#([^\s]+)', '', text)
        text = self.decontracting(text)
        text = word_tokenize(text)
        return [self.english_stemmer.stem(word) for word in text if word not in self._stopwords]

    def building_vocab(self, data):
        all_words = []

        for sentence, emotion in data:
            all_words.extend(sentence)

        # counts frequency of words
        word_list = nltk.FreqDist(all_words)

        self.word_freqs = {k: v for k, v in sorted(word_list.items(), key=lambda item: item[1], reverse=True)}
        self.vocab = list(self.word_freqs.keys())
        self.vocab.append('<UNK>')

    def preprocessing_train(self, total):
        ''' taking input to go through all preprocessing steps
        also builds vocabulary for further steps '''
        processed = []

        for sentence, emotion in total:
            tmp = self.cleaning(sentence.strip())
            if len(tmp) != 0:
                processed.append([tmp, emotion])

        self.building_vocab(processed)

        return processed

    def preprocessing_test(self, total):
        ''' preprocessing for test data set all the way to encoding '''
        processed = []

        for sentence in total:
            tmp = self.cleaning(sentence.strip())
            if len(tmp) != 0:
                processed.append(self.text_to_encoding(tmp))

        return processed


    def text_to_encoding(self, sentence):
        ''' encoding text sentences to vocab encoded integer list '''
        encoded = []

        for word in sentence:
            if word in self.vocab:
                encoded.append(self.vocab.index(word) + 1)
            else:
                encoded.append(len(self.vocab))

        return encoded

    def output_csv(self, sentences, labels, probabilities, file_name):
        ''' creating csv files for each sentences and
        the expected classification output(label and probability) '''

        output = pd.DataFrame({'sentence': sentences,
                               'label': labels,
                               'probability': probabilities})

        return output.to_csv('../../Output/'+file_name+'_transformer.csv',
                             sep=';',
                             columns=['sentence', 'label', 'probability'],
                             index=False)
