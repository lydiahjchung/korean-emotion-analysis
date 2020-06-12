from nltk.stem.snowball import SnowballStemmer
from string import punctuation 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import re
import keras
import nltk

class PreProcess:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')

        self.english_stemmer = SnowballStemmer('english')
        temps = ['``', "''", '…', '—', '~~', '"', '..', '“', '-_____-', 'm̶̲̅ε̲̣', 'rt', '=/', '»']
        itos = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self._stopwords = set(stopwords.words('english') + list(punctuation) + list(range(11)) + temps + itos)

    # Vocabulary 만들기
    def build_vocab(self, data):
        all_words = []

        for words, emotion in data:
            all_words.extend(words)

        wordlist = nltk.FreqDist(all_words)
        word_features = wordlist.keys()
        return wordlist, word_features

    # 데이터 정규화 및 정제를 위한 함수
    def decontract(self, phrase):
        # 데이터 정제(Cleaning)
        phrase = re.sub(r"http\\S+", "", phrase)                           # 하이퍼링크 제거
        phrase = re.sub('((www\\.[^\\s]+)|(https?://[^\\s]+))', '', phrase)  # 트위터링크 제거
        phrase = re.sub('@[^\\s]+', '', phrase)                           # 트위터아이디 제거

        # 데이터 정규화(Normalization)
        phrase = re.sub('#([^\\s]+)', '', phrase)
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

    def pre_to_tok(self, text):
        text = self.decontract(text)     # 데이터 정제 및 정규화
        text = word_tokenize(text)  # 데이터 토큰화
        return [self.english_stemmer.stem(word) for word in text if word not in self._stopwords]  # 어간 추출 및 불용어 처리

    def test_preprocess(self, platform, keys_sorted, max_len):
        with open("/Data/{}_final_test.txt".format(platform)) as f:
            testdata = f.readlines()

        final_test, processed_test = [], []

        # 데이터 한 줄씩 읽기
        for line in testdata:
            final_test.append(line.strip())

        # 데이터 전처리
        for each in final_test:
            done = self.pre_to_tok(each)
            if len(done) != 0:
                processed_test.append(done)

        # 데이터 토큰화 및 패딩
        test_vect = []
        for sent in processed_test:
            vect = []
            for word in sent:
                if word in keys_sorted:
                    vect.append(keys_sorted.index(word)+1)
                else:
                    vect.append(len(keys_sorted))
            test_vect.append(vect)
        test_vect = keras.preprocessing.sequence.pad_sequences(test_vect, maxlen=max_len)
        return test_vect, final_test