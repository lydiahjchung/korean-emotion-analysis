# korean-emotion-analysis
<div align="center">
  <img src="https://user-images.githubusercontent.com/38775259/81778498-f2ee6c80-952d-11ea-91f3-347ce1ffcadd.jpg" width="600", height="400"></img>
</div>

--------------
## Introduction
**Single-polarity sentiment analysis**는 문장의 감정을 **긍정/부정/중립**으로 나눠서 분석한다. 
하지만 부정적인 감정에는 공포와 불안, 불안 등 여러 가지 감정이 있듯이 이 모두를 부정적인 감정이라고 여기기에는 빠지는 정보가 많다. 
영어의 경우 그 단점을 보완하기 위해 감정을 단순히 긍정, 부정으로 나누는 게 아닌, 더 상세하게 나누는 연구가 활발히 진행 중이다. 
트위터, 댓글, 온라인 뉴스 데이터를 사용하여 **‘happy, sad, anger, disgust, surprise, fear’** 등과 같은 세부적인 감정들로 감정 카테고리를 분류하고 있다.
<br><br>
한국어 문장은 영어와는 다르게 형태소별로 나눌 수 있는 교착어로 이루어져 있으며 이 특성으로 인해 한국어로 된 감성 사전 구축이 상대적으로 더 어렵다.
감성 사전 구축의 어려움으로 인하여 한국어 **sentiment-analysis** 또한 **single-polarity**에 머물고 있다. 
여러 카테고리의 감성 분석 결과를 활용하여 여론조사를 대체하고 있는 미국과는 다르게 한국은 한정적인 분야에서만 감성 분석 결과를 사용하고 있다. 
<br><br>
한국어 문장을 emotion analysis를 통해 **7가지 감정 카테고리(happy, sad, anger, disgust, surprise, fear, neutral) classification**을 진행하는 것을 목표로 한다. 
한국어 데이터 수집하여 **구글, 파파고, 카카오 NMT API**를 통해 **영문 번역 데이터**를 제작하고 그 결과를 모델에 대입해본다. 
**mLSTM, Transformer, Multi-class SVM kernels** 모델에 이미 공개된 labeled 된 영문 데이터를 training data를 사용하여 multi-category classification emotion analysis 모델을 구축한다.
이후 총 아홉 가지의 데이터-모델 조합의 output을 비교하고, 최적의 조합을 제시한다.

## Content
1. [Introduction](#introduction)
2. [데이터 크롤링](#데이터-크롤링)
    * [Twitter 데이터 크롤링](#twitter-데이터-크롤링)
    * [Twitter 데이터 전처리](#twitter-데이터-전처리)
3. [NMT API를 사용한 크롤링 데이터 번역](#nmt-api를-사용한-크롤링-데이터-번역)
    * [Google NMT API](#google-nmt-api)
    * [Naver Papago NMT API](-#aver-papago-nmt-api)
    * [Kakao NMT API](#kakao-nmt-api)
4. [감성 분석 모델](#감성-분석-모델)
    * [mLSTM + attention](#mlstm-+-attention)
    * [Transformer](#transformer)
    * [Multiclass SVM](#multiclass-svm)
5. [결과 분석](#결과-분석)

## 데이터 크롤링
- **Twitter 데이터 크롤링**<br>
    Tweeter Crawling API로 TWEEPY가 있으나 최근 7일 데이터만 수집할 수 있는 한계가 있다.<br>
    그 이전의 데이터를 수집하고 싶으면 Premium-Api를 구매해야 하는데 500request에 $149/월 이다.<br>
    따라서 오픈소스로 많이 사용하는 twitterscraper package를 사용하려고 한다.
  ```python
    try:
        from twitterscraper.query import query_tweets
        from twitterscraper.tweet import Tweet
    except:
        !pip install twitterscraper
        from twitterscraper.query import query_tweets
        from twitterscr  ```
  데이터는 **총선**을 키워드로 검색하였다.
  ```python
  list_of_tweets = query_tweets('총선', begindate=datetime.date(2020,4,1), 
         aper.tweet import Tweet
                       enddate=datetime.date(2020,4,30))
  ```
- **Twitter 데이터 전처리**<br>
    데이터 전처리를 위하여 [**Soyspacing**](https://github.com/lovit/soyspacing) 패키지를 사용하였다. 추가적으로 링크, 트위터 아이디 등을 불용어처리 하였다.
  ```python
    remove_hypterlink = re.sub(r"http\S+", "", sentence['content'])       # 하이퍼링크 제거
    remove_twitterlink = re.sub(r"pic\S+", "", remove_hypterlink)         # 트위터링크 제거
    remove_retweet = re.sub(r"@\S+", "", remove_twitterlink)              # 트위터아이디 제거
  ```
- **영화 데이터 크롤링**<br>
    영화 데이터는 [**NSMC**](https://github.com/e9t/nsmc)의 네이버 영화 리뷰 데이터를 긍정/부정으로 분류한 다음의 자료를 활용하였다.<br>
    
- **영화 데이터 전처리**<br>
  긍정과 부정으로 레이블링 되어있는 것을 제거하고 자음, 특수문자 그리고 불필요한 공백을 제거하였다. 
  ```python
    #문자마다 마지막에 있는 0,1 값 제거 
    data_train = [line.strip('0') for line in data_train]
    data_train = [line.strip('1') for line in data_train]
    data_train = [line.strip('\t') for line in data_train]
    data_train = [line.replace('\t',' ' ) for line in data_train]

    #자음 제거
    data_train = [line.replace('ㅋ','' ) for line in data_train]
    data_train = [line.replace('ㅜ','' ) for line in data_train]
    data_train = [line.replace('ㅠ','' ) for line in data_train]
    data_train = [line.replace('ㅎ','' ) for line in data_train]
    data_train = [line.replace('ㄱ','' ) for line in data_train]
    data_train = [line.replace('ㅉ','' ) for line in data_train]
    data_train = [line.replace('ㅅ','' ) for line in data_train]
    data_train = [line.replace('ㅂ','' ) for line in data_train]
    data_train = [line.replace('ㅈ','' ) for line in data_train]
    data_train = [line.replace('ㅊ','' ) for line in data_train]
    data_train = [line.replace('ㅊ','' ) for line in data_train]
    data_train = [line.replace('ㅏ','' ) for line in data_train]

    #특수 문자 제거
    data_train = [line.replace('*','' ) for line in data_train]
    data_train = [line.replace(';','' ) for line in data_train]
    data_train = [line.replace('♥','' ) for line in data_train]
    data_train = [line.replace('/','' ) for line in data_train]
    data_train = [line.replace('♡','' ) for line in data_train]
    data_train = [line.replace('>','' ) for line in data_train]
    data_train = [line.replace('<','' ) for line in data_train]
    data_train = [line.replace('-','' ) for line in data_train]
    data_train = [line.replace('_','' ) for line in data_train]
    data_train = [line.replace('+','' ) for line in data_train]
    data_train = [line.replace('=','' ) for line in data_train]
    data_train = [line.replace('"','' ) for line in data_train]
    data_train = [line.replace('~','' ) for line in data_train]
    data_train = [line.replace('^','' ) for line in data_train]

    #숫자 제거
    data_train = [line.replace('0','' ) for line in data_train]
    data_train = [line.replace('1','' ) for line in data_train]
    data_train = [line.replace('2','' ) for line in data_train]
    data_train = [line.replace('3','' ) for line in data_train]
    data_train = [line.replace('4','' ) for line in data_train]
    data_train = [line.replace('5','' ) for line in data_train]
    data_train = [line.replace('6','' ) for line in data_train]
    data_train = [line.replace('7','' ) for line in data_train]
    data_train = [line.replace('8','' ) for line in data_train]
    data_train = [line.replace('9','' ) for line in data_train]

    #왼쪽 공백 제거
    data_train = [line.lstrip( ) for line in data_train]

    #오른쪽 공백 제거
    data_train = [line.rstrip( ) for line in data_train]

  ```
## NMT API를 사용한 크롤링 데이터 번역
- **Google NMT API**<br>
  Google NMT API는 **The Python Package Index(PyPI)** 에 올라와 있는 [**공식 API 사용 예제**](https://pypi.org/project/googletrans/)에 따라 구현하였다.
```python
from googletrans import Translator

for i in range(len(data_train)):
  string = data_train[i]
  translator = Translator(proxies=None, timeout=None)
  result = translator.translate(string, dest="en")
  print(result.text)
```
- **Naver Papago NMT API**<br>
  Papago NMT API는 **Naver Developers**에 올라와 있는 [**공식 API 사용 예제**](https://developers.naver.com/docs/nmt/reference)에 따라 구현하였다.<br>
  이후 결과를 **JSON**파일로 저장하여 **Input 데이터**로 사용하기 쉽게 저장하였다.
```python
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
```
- **Kakao NMT API**<br>
  [**공식 번역 개발 가이드**](https://developers.kakao.com/docs/latest/ko/translate/dev-guide)를 참고하여 Kakao NMT API를 구현하였다. <br>
```python
URL = 'https://kapi.kakao.com/v1/translation/translate'
APP_KEY = {APP KEY}

r = requests.get(URL, headers=headers, params = paras )
json_data = json.loads(r.text)
trans_text = json_data.get('translated_text')
 ```
-------------
## 감성 분석 모델
- [**mLSTM + attention**](Model/mlstm_attention/mlstm_attention.md)
- [**Transformer**](Model/transformer/transformer.md)
- [**Multiclass SVM**](Model/msvm_kernel/msvm_kernel.md)

## 모델 분석 결과
### NMT API별 모델 분석 결과
각 NMT API별로 MSVM, mLSTM Attention, Transformer를 조합하여 분석한 결과의 상위 24개 데이터이다.
또한 Movie 데이터와 Twitter 데이터에 두드러지는 차이점이 존재하기 때문에 Movie 데이터만 사용했을 때의 Accuracy와 Twitter 데이터만 사용했을 때의 Accuracy, 그리고 두 데이터를 모두 사용했을 때의 Accuracy를 모두 측정하였다.
<div align="center">
  <img src="https://user-images.githubusercontent.com/38775259/84489650-f7e53e00-acdc-11ea-824c-562be1a4a2b9.png" width="1000", height="400"></img>
</div><br>

<html lang="ko">
  <head>
    <meta charset="utf-8">
  </head>
  <body>
    <table>
        <tr>
          <th colspan="1" rowspan="2" width="10%">라벨</th>
          <th colspan="3" width="30%">구글</th>
          <th colspan="3" width="30%">카카오</th>
          <th colspan="3" width="30%">파파고</th>
        </tr>
        <tr>
          <th width="10%">SVM</th>
          <th width="10%">MLSTM</th>
          <th width="10%">TF</th>
          <th width="10%">SVM</th>
          <th width="10%">MLSTM</th>
          <th width="10%">TF</th>
          <th width="10%">SVM</th>
          <th width="10%">MLSTM</th>
          <th width="10%">TF</th>
        </tr>
        <tr>
          <th>Movie</th>
          <td align="center">35</td>
          <td align="center">39</td>
          <td align="center">32</td>
          <td align="center">31</td>
          <td align="center">39</td>
          <td align="center">29</td>
          <td align="center">34</td>
          <td align="center">36</td>
          <td align="center">27</td>
        </tr>
        <tr>
          <th>Twitter</th>
          <td align="center">57</td>
          <td align="center">34</td>
          <td align="center">24</td>
          <td align="center">57</td>
          <td align="center">32</td>
          <td align="center">16</td>
          <td align="center">51</td>
          <td align="center">37</td>
          <td align="center">16</td>
        </tr>
        <tr>
          <th>Total</th>
          <td align="center">92</td>
          <td align="center">73</td>
          <td align="center">56</td>
          <td align="center">88</td>
          <td align="center">71</td>
          <td align="center">45</td>
          <td align="center">85</td>
          <td align="center">73</td>
          <td align="center">43</td>
        </tr>
        <tr>
          <th>M_Accuracy</th>
          <td align="center">0.35</td>
          <td align="center">0.39</td>
          <td align="center">0.32</td>
          <td align="center">0.31</td>
          <td align="center">0.39</td>
          <td align="center">0.29</td>
          <td align="center">0.34</td>
          <td align="center">0.36</td>
          <td align="center">0.27</td>
        </tr>
        <tr>
          <th>T_Accuracy</th>
          <td align="center">0.57</td>
          <td align="center">0.34</td>
          <td align="center">0.24</td>
          <td align="center">0.57</td>
          <td align="center">0.32</td>
          <td align="center">0.16</td>
          <td align="center">0.51</td>
          <td align="center">0.37</td>
          <td align="center">0.16</td>
        </tr>
        <tr>
          <th>Total Accuracy</th>
          <td align="center">0.46</td>
          <td align="center">0.365</td>
          <td align="center">0.28</td>
          <td align="center">0.44</td>
          <td align="center">0.355</td>
          <td align="center">0.225</td>
          <td align="center">0.425</td>
          <td align="center">0.365</td>
          <td align="center">0.215</td>
        </tr>
    </table>
  </body>
</html>

- **구글 NMT API + Model Accuracy**<br>
  - 영화 데이터: mLSTM + Attention - 0.39
  - 트위터 데이터: mSVM - 0.39
  - 모든 데이터: mSVM - 0.36
  
- **카카오 NMT API + Model Accuracy**<br>
  - 영화 데이터: mLSTM + Attention - 0.39
  - 트위터 데이터: mSVM - 0.57
  - 모든 데이터: mSVM - 0.44
  
- **파파고 NMT API + Model Accuracy**<br>
  - 영화 데이터: mLSTM + Attention - 0.36
  - 트위터 데이터: mSVM - 0.51
  - 모든 데이터: mSVM - 0.43

### 모델별 NMT API 분석 결과
각 모델별로 Google, Kakao, Papago NMT API를 조합하여 분석한 결과의 상위 24개 데이터이다.
<div align="center">
  <img src="https://user-images.githubusercontent.com/38775259/84489534-ce2c1700-acdc-11ea-8efb-29ca974db2ed.png" width="1000", height="400"></img>
</div><br>

<html lang="ko">
  <head>
    <meta charset="utf-8">
  </head>
  <body>
    <table>
        <tr>
          <th colspan="1" rowspan="2" width="10%">라벨</th>
          <th colspan="3" width="30%">SVM</th>
          <th colspan="3" width="30%">MLSTM</th>
          <th colspan="3" width="30%">TF</th>
        </tr>
        <tr>
          <th width="10%">구글</th>
          <th width="10%">카카오</th>
          <th width="10%">파파고</th>
          <th width="10%">구글</th>
          <th width="10%">카카오</th>
          <th width="10%">파파고</th>
          <th width="10%">구글</th>
          <th width="10%">카카오</th>
          <th width="10%">파파고</th>
        </tr>
        <tr>
          <th>Movie</th>
          <td align="center">35</td>
          <td align="center">31</td>
          <td align="center">34</td>
          <td align="center">39</td>
          <td align="center">39</td>
          <td align="center">36</td>
          <td align="center">32</td>
          <td align="center">29</td>
          <td align="center">27</td>
        </tr>
        <tr>
          <th>Twitter</th>
          <td align="center">57</td>
          <td align="center">57</td>
          <td align="center">25</td>
          <td align="center">34</td>
          <td align="center">32</td>
          <td align="center">37</td>
          <td align="center">24</td>
          <td align="center">16</td>
          <td align="center">16</td>
        </tr>
        <tr>
          <th>Total</th>
          <td align="center">92</td>
          <td align="center">88</td>
          <td align="center">85</td>
          <td align="center">73</td>
          <td align="center">71</td>
          <td align="center">73</td>
          <td align="center">56</td>
          <td align="center">45</td>
          <td align="center">43</td>
        </tr>
        <tr>
          <th>M_Accuracy</th>
          <td align="center">0.35</td>
          <td align="center">0.31</td>
          <td align="center">0.34</td>
          <td align="center">0.39</td>
          <td align="center">0.39</td>
          <td align="center">0.36</td>
          <td align="center">0.32</td>
          <td align="center">0.29</td>
          <td align="center">0.27</td>
        </tr>
        <tr>
          <th>T_Accuracy</th>
          <td align="center">0.57</td>
          <td align="center">0.57</td>
          <td align="center">0.25</td>
          <td align="center">0.34</td>
          <td align="center">0.32</td>
          <td align="center">0.37</td>
          <td align="center">0.24</td>
          <td align="center">0.16</td>
          <td align="center">0.16</td>
        </tr>
        <tr>
          <th>Total Accuracy</th>
          <td align="center">0.46</td>
          <td align="center">0.44</td>
          <td align="center">0.425</td>
          <td align="center">0.365</td>
          <td align="center">0.355</td>
          <td align="center">0.365</td>
          <td align="center">0.28</td>
          <td align="center">0.225</td>
          <td align="center">0.215</td>
        </tr>
    </table>
  </body>
</html>

- **mSVM + NMT API Accuracy**<br>
  - 영화 데이터: 구글 NMT API - 0.35
  - 트위터 데이터: 구글 or 카카오 NMT API - 0.57
  - 모든 데이터: 구글 NMT API - 0.46
  
- **mLSTM Attention + NMT API Accuracy**<br>
  - 영화 데이터: 구글 or 카카오 NMT API - 0.39
  - 트위터 데이터: 파파고 NMT API - 0.37
  - 모든 데이터: 구글 or 카카오 NMT API - 0.37
  
- **Transformer + NMT API Accuracy**<br>
  - 영화 데이터: 구글 NMT API - 0.32
  - 트위터 데이터: 구글 NMT API - 0.24
  - 모든 데이터: 구글 NMT API - 0.28
  
### NMT API별 모델 분석 총평 
- 영화 데이터: 구글 or 카카오 NMT API + mLSTM Attention - 0.39
- 트위터 데이터: 구글 or 카카오 NMT API + SVM - 0.57
- 총 데이터: 구글 NMT API + SVM - 0.46

따라서 영화 리뷰 데이터와 같이 문장이 짧고 감정이 다양한 Input Data를 사용할 경우에는 구글 또는 카카오 NPT API와 mLSTM + Attention 모델 조합을 사용하는 것이 성능이 가장 우수하다는 것을 알 수 있다. 반면 트위터 데이터와 같이 문장이 대체적으로 길고 감정이 한정적인 Input Data를 사용할 경우에는 구글 또는 카카오 NPT API와 mSVM 모델 조합을을 사용하는 것이 가장 우수하다. 마지막으로 전체 데이터를 사용했을 때는 구글 데이터와 mSVM 모델 조합을 사용했을 경우 0.43으로 성능이 가장 우수하였다.
