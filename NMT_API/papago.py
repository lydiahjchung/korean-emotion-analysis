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
with open('Crawler\\preprocessed_files\\pre_data_0401_0402.txt', 'r', encoding='utf-8') as f:
    raw_sentence = f.read()

# API Key
client_id = "NJGTWFNflleFvDR2wvqu" 
client_secret = "Type Password" 

# Naver Papago Open API
encText = urllib.parse.quote(raw_sentence)
data = "source=ko&target=en&text=" + encText
url = "https://openapi.naver.com/v1/papago/n2mt"
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()

if(rescode==200):
    response_body = response.read()
    
    # Json format
    result = json.loads(response_body.decode('utf-8'))
    pprint(result)

    # Json result  
    with open('Crawler\\translated_files\\translated_0401_0402.txt', 'w', encoding='utf8') as f:
        f. write(result['message']['result']['translatedText'])
else:
    print("Error Code:" + rescode)