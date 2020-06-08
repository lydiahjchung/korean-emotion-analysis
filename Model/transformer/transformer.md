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

    import re
    import nltk
    from nltk.tokenize import word_tokenize
    from string import punctuation
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download('punkt')
    nltk.download('stopwords')

#### Loading the data

    # loading the original labeled data
    with open("labeled_final.txt") as f:
      data = f.readlines()

    # labeled = [[sentence, emotion], [sentence, emotion], ... ]
    labeled = []
    for line in data:
      cut_idx = len(line) - line[::-1].find(";")
      sent, em = line[:cut_idx-1], line[cut_idx:].strip()
      labeled.append([sent, em])
      
#### Cleaning and Normalizing

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
