import re
import datetime
import requests
from urllib.parse import urlparse
import json

# Kakao Translation NMT API
URL = 'https://kapi.kakao.com/v1/translation/translate'
# Kakao APP KEY to use REST API
APP_KEY = 'e854c39af6f0eb88b4d053085692d549'

# Using NMT API
def translate(type):
    tmp = []
    for each in type:
        query = each
        headers = {'Authorization': 'KakaoAK {}'.format(APP_KEY)}
        paras = {"query":query, "target_lang":"en", "src_lang":"kr"}

        r = requests.get(URL, headers=headers, params = paras )
        json_data = json.loads(r.text)
        trans_text = json_data.get('translated_text')
        tmp.append(trans_text)
    return tmp

# opening original data
with open('../Data/movie_100.txt') as f:
    movie_100 = f.readlines()
with open('../Data/twitter_100.txt') as f:
    twitter_100 = f.readlines()

movie, twitter = [], []
for line in movie_100:
    movie.append(line.strip())
for line in twitter_100:
    twitter.append(line.strip())

# translating movie and twitter data
movies = translate(movie)
twitters = translate(twitter)

# saving final data set
with open('../Data/kakao_final_test.txt', 'w') as f:
    for each in movies:
        try:
          line = each[0][0]
          f.writelines(line + '\n')
        except TypeError:
          continue
    for each in twitters:
        try:
          line = each[0][0]
          f.writelines(line + '\n')
        except TypeError:
          continue
