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
    ```
    try:
        from twitterscraper.query import query_tweets
        from twitterscraper.tweet import Tweet
    except:
        !pip install twitterscraper
        from twitterscraper.query import query_tweets
        from twitterscraper.tweet import Tweet
    ```
    데이터는 **'총선'**을 키워드로 검색하였다.
    ```
    list_of_tweets = query_tweets('총선', begindate=datetime.date(2020,4,1), 
                                enddate=datetime.date(2020,4,30))
    ```
- **Twitter 데이터 전처리**<br>
    데이터 전처리를 위하여 [**Soyspacing**](https://github.com/lovit/soyspacing) 패키지를 사용하였다. 추가적으로 링크, 트위터 아이디 등을 불용어처리 하였다.
    ```
      remove_hypterlink = re.sub(r"http\S+", "", sentence['content'])       # 하이퍼링크 제거
      remove_twitterlink = re.sub(r"pic\S+", "", remove_hypterlink)         # 트위터링크 제거
      remove_retweet = re.sub(r"@\S+", "", remove_twitterlink)              # 트위터아이디 제거
    ```

## NMT API를 사용한 크롤링 데이터 번역
- **Google NMT API**<br>
  Google NMT API는 **The Python Package Index(PyPI)** 에 올라와 있는 [**공식 API 사용 예제**](https://pypi.org/project/googletrans/)에 따라 구현하였다.
    ```
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
  ```
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
  ```
  URL = 'https://kapi.kakao.com/v1/translation/translate'
  APP_KEY = {APP KEY}

  r = requests.get(URL, headers=headers, params = paras )
  json_data = json.loads(r.text)
  trans_text = json_data.get('translated_text')
   ```
-------------
## 감성 분석 모델
- **mLSTM + attention**
- **Transformer**
- **Multiclass SVM**<br>
  ```
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
  import matplotlib.pyplot as plt
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.feature_extraction.text import TfidfTransformer
  from sklearn.linear_model import SGDClassifier
  from sklearn.model_selection import cross_val_score, GridSearchCV
  
  df = pd.read_csv('/content/drive/Shared drives/데이터분석캡스톤디자인/데이터/라벨 데이터/labeled_final.txt',sep=';', names =['text','label'])
  X = df.text
  y = df.label
  my_tags = ['anger','happiness','surprise', 'sadness', 'fear', 'neutral', 'disgust']
  
  cvect = CountVectorizer()

  cvect_X_train = cvect.fit(X_train)
  cvect_X_test = cvect.fit(X_test)
  cvect.vocabulary_

  cvect_X_train = cvect.transform(X_train).toarray()
  cvect_X_test = cvect.transform(X_test).toarray()

  tfid = TfidfTransformer()

  tt_X_train = tfid.fit(cvect_X_train)
  tt_X_test = tfid.fit(cvect_X_test)

  tt_X_train = tfid.transform(cvect_X_train).toarray()
  tt_X_test = tfid.transform(cvect_X_test).toarray()

  from sklearn.calibration import CalibratedClassifierCV

  sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-2, random_state=42, max_iter=10, tol=None)
  sgd = sgd.fit(tt_X_train, y_train)

  y_pred = sgd.predict(tt_X_test)

  clf = CalibratedClassifierCV(sgd) 
  clf.fit(tt_X_train, y_train)

  y_proba = clf.predict_proba(tt_X_test)

  print(clf.classes_)
  
  ```
## 모델 분석 결과
- **mLSTM + attention**
- **Transformer**
- **Multiclass SVM**<br>
  ```
  #Accuracy Score 확인
  print("train accuracy:", sgd.fit(X_train, y_train).score(X_train, y_train))
  print('validation accuracy: %s' % accuracy_score(y_pred, y_test))
  print(classification_report(y_test, y_pred,target_names=my_tags))

  #확률 예측 값 & 예측 클래스 확인
  X_test = list(X_test)
  X_test
  i = 0
  
  for i in range(len(X_test)):
    score = np.max(y_proba[i,:])*100

    idx = np.argmax(y_proba[i,:])
    emotion = clf.classes_[idx]

    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))
  ```
