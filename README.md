# korean-emotion-analysis
<div align="center">
  <img src="https://user-images.githubusercontent.com/38775259/81778498-f2ee6c80-952d-11ea-91f3-347ce1ffcadd.jpg" width="600", height="400"></img>
</div>

--------------
## Introduction
**Single-polarity sentiment analysis**는 문장의 감정을 **‘긍정/부정/중립’**으로 나눠서 분석한다. 
하지만 부정적인 감정에는 공포와 불안, 불안 등 여러 가지 감정이 있듯이 이 모두를 부정적인 감정이라고 여기기에는 빠지는 정보가 많다. 
영어의 경우 그 단점을 보완하기 위해 감정을 단순히 긍정, 부정으로 나누는 게 아닌, 더 상세하게 나누는 연구가 활발히 진행 중이다. 
트위터, 댓글, 온라인 뉴스 데이터를 사용하여 **‘happy, sad, anger, disgust, surprise, fear’** 등과 같은 세부적인 감정들로 감정 카테고리를 분류하고 있다.
<br><br>
한국어 문장은 영어와는 다르게 형태소별로 나눌 수 있는 교착어로 이루어져 있으며 이 특성으로 인해 한국어로 된 감성 사전 구축이 상대적으로 더 어렵다.
감성 사전 구축의 어려움으로 인하여 한국어 **sentiment-analysis** 또한 **single-polarity**에 머물고 있다. 
여러 카테고리의 감성 분석 결과를 활용하여 여론조사를 대체하고 있는 미국과는 다르게 한국은 한정적인 분야에서만 감성 분석 결과를 사용하고 있다. 
<br><br>
한국어 문장을 emotion analysis를 통해 **7가지 감정 카테고리(happy, sad, anger, disgust, surprise, fear, neutral) classification**을 진행하는 것을 목표로 한다. 
한국어 데이터 수집하여 **세 가지 NMT API(구글, 파파고, 카카오)**를 통해 **영문 번역 데이터**를 제작하고 그 결과를 모델에 대입해본다. 
**mLSTM, Transformer, Multi-class SVM kernels** 모델에 이미 공개된 labeled 된 영문 데이터를 training data를 사용하여 multi-category classification emotion analysis 모델을 구축한다.
이후 총 아홉 가지의 데이터-모델 조합의 output을 비교하고, 최적의 조합을 제시한다.

## 데이터 크롤링
- **Twitter 데이터 크롤링**<br>
Tweeter Crawling API로 TWEEPY가 있나 최근 7일 데이터만 수집할 수 있는 한계가 있다.<br>
그 이전의 데이터를 수집하고 싶으면 Premium-Api를 구매해야 하는데 500request에 $149/월 이다.<br>
따라서 오픈소스로 많이 사용하는 twitterscraper package를 사용하려고 한다.
## NMT API를 사용한 크롤링 데이터 번역
- **Google NMT API**
- **Naver PaPago NMT API**
- **Kakao NMT API**
