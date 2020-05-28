"""
네이버 파파고 API
처리한도: 하루 10,000개 
"""
import os
import sys
import urllib.request
import json
from pprint import pprint

# Read sentence you want to translate
f = open("twittter_pre_data.txt", 'r', encoding='utf-8')
raw_sentence = f.readlines()
sentence_list = raw_sentence
sentence_list = sentence_list[241:]
# API Key
client_id = "NJGTWFNflleFvDR2wvqu"
client_secret = "temp"

with open('papago_twitter.txt', 'w', encoding='utf8') as f:
    count = 1
    for sentence in sentence_list:
        encText = urllib.parse.quote(sentence)
        data = "source=ko&target=en&text=" + encText
        url = "https://openapi.naver.com/v1/papago/n2mt"
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
        rescode = response.getcode()

        if(rescode == 200):
            response_body = response.read()

            # Json format
            result = json.loads(response_body.decode('utf-8'))

            # Json result
            f. write(result['message']['result']['translatedText'] + "\n")
            print("Translated Completed #{}".format(count))
            count += 1
        else:
            print("Error Code:" + rescode)
