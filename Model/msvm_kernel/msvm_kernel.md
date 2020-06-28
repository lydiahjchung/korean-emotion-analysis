## 감정 데이터셋
모델 학습을 위해 **6가지 카테고리의 감정 데이터**가 라벨링된 데이터셋을 사용하려고 한다.<br>
6가지 감정 카테고리는 **anger, happiness, neutral, surpirse, sadness, fear, dissgust**로 이루어져 있다.

```python
    import numpy as np 
    import pandas as pd 
    import os.path

    path = os.getcwd()
    col_name = ['text', 'emotions']

    # 데이터 셋 불러오기
    dataset = pd.read_csv(path + "/labeled_final.txt", names=col_name, sep=';')

    dataset
    # 라벨 데이터 히스토그램
    dataset.emotions.value_counts().plot.bar(align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan', "purple"])
```
<div align="center">
  <img src="https://user-images.githubusercontent.com/38775259/84282583-e54ff500-ab74-11ea-9fdb-87b7a3d29583.png" width="400", height="300"></img>
</div>

## Train Set 불러오기
라벨링된 데이터를 **pd.read_csv**을 통해 불러온다.

```python
    import os
    import logging
    from numpy import random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC, LinearSVC #squared hinge loss
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import SGDClassifier


    path = os.getcwd()
    col_name = ['text', 'label']

    #trainset 불러오기
    df_train = pd.read_csv(path + "/labeled_final.txt", names=col_name, sep=';')
    X_train = df_train.text
    y_train = df_train.label
```

## 데이터 Preprocessing
최종적으로 twitter dataset의 특수문자를 제거하였다.

```python
    df_X = list(X_train)

    if any("@" in s for s in df_X):
        print("있음")
    else:
      print("없음")

    df_text = list(X_train)
    len(df_text)

    df_X = [line.replace('@','' ) for line in df_X]
    df_X = [line.replace('m̶̲̅ε̲̣','' ) for line in df_X]
    df_X = [line.replace('...','' ) for line in df_X]
    df_X = [line.replace("''",'' ) for line in df_X]
    df_X = [line.replace('~','' ) for line in df_X]
    df_X = [line.replace('``','' ) for line in df_X]
    df_X = [line.replace('—','' ) for line in df_X]
    df_X = [line.replace('"','' ) for line in df_X]
    df_X = [line.replace('..','' ) for line in df_X]
    df_X = [line.replace('“','' ) for line in df_X]
    df_X = [line.replace('-_____-','' ) for line in df_X]
    df_X = [line.replace('rt','' ) for line in df_X]
    df_X = [line.replace('=/','' ) for line in df_X]
    df_X = [line.replace('»','' ) for line in df_X]
    df_X = [line.replace('http','' ) for line in df_X]

    if any("@" in s for s in df_X):
        print("있음")
    else:
      print("없음")

    #trainset 최종 전처리
    df_train['text'] = df_X

    X_train = df_train.text
    y_train = df_train.label
```

## 라벨 시각화
레이블 분포가 균일한지 시각화를 통해 확인하였다.
```python
    #레이블 분포 확인
    df_train['label'].value_counts()

    #레이블 분포 시각화
    my_tags = ['anger','happiness','surprise', 'sadness', 'fear', 'neutral', 'disgust']
    plt.figure(figsize=(10,4))
    y_train.value_counts().plot(kind='bar');
```

## Test Set 불러오기
파파고, 카카오, 구글 데이터 셋을 불러온다.
```python
    # Test set 불러오기
    #Google
    df_test = pd.read_csv('/content/drive/Shared drives/데이터분석캡스톤디자인/데이터/최종 test 데이터/google_final_test.txt',sep=';', names =['text','label'])
    X_test = df_test.text
    y_test = df_test.label

    #Kakao
    df_test = pd.read_csv('/content/drive/Shared drives/데이터분석캡스톤디자인/데이터/최종 test 데이터/kakao_final_test.txt',sep=';', names =['text','label'])
    X_test = df_test.text
    y_test = df_test.label

    #Papago
    twitter = []        # 트위터 데이터 초기화
    label = []          # 라벨링 데이터 초기화

    # 데이터 불러오기
    with open('/content/drive/Shared drives/데이터분석캡스톤디자인/데이터/최종 test 데이터/papago_final_test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

    # 트위터/라벨링 데이터 리스트로 저장하기
    for line in lines:
      twitter.append(''.join(line.split(';')[0:-1]).replace('\n', ''))  # 트위터 데이터
      label.append(line.split(';')[-1].replace('\n', ''))               # 라벨링 데이터

    X_test = pd.Series(twitter) 
    y_test = pd.Series(label)

    print(X_test)
    print(type(X_test))
```

## 모델 훈련
**데이터 전처리:**<br>
 mSVM 모델에 적용하기 전에 두 단계를 거쳐 전처리를 진행하였다. 가장 먼저, 텍스트를 토큰 리스트로 변환한 후 각 텍스트에서 토큰 출현 빈도를 센 다음 각 텍스트를 BOW 인코딩 벡터로 변환하는 CountVectorizer를 사용하였다. 이후 CountVectorizer를 사용하여 단어 수를 계산한 뒤 IDF(역 문서 빈도)값 계산한 후 Tf-idf 점수 계산하는 Tfidftransformer를 사용하여 노멀라이제이션을 진행하였다.<br>
**모델 선정 이유:**<br>
 N. M. Hakak, M. Mohd, M. Kirmani and M. Mohd, “Emotion analysis: A survey” 에서는 감정 분석과 관련하여 감정의 표현, 여러가지 감정 분석 모델, 감정 분석 모델에 적용된 데이터, 여러가지 computational 접근법을 소개하고 있다. 이때 각 감정 분석 모델의 성능을 비교한 부분이 있는데 Balabantaray, Mudasir Mohammad, and Nibha Sharma- Multi-Class Twitter Emotion Classification: A New Approach 아티클에서 사용한 Multi Class SVM의 정확도가 73.24%로 가장 높은 것을 확인할 수 있었다. 따라서 해당 모델을 사용하여 감정 분석을 진행하였다.<br>
**모델 특징:**<br>
 KSVM은 SVM과는 다르게 선형적으로 분류되지 않는 클래스를 구분하도록 비선형적 특성을 추가한 모델이다. SVM에 Kernel Trick을 사용하여 데이터를 확장하지 않고 확장된 특성에 대한 데이터 포인트들의 거리를 계산한다. 해당 모델을 사용한 Balabantaray, Mudasir Mohammad, and Nibha Sharma- Multi-Class Twitter Emotion Classification: A New Approach에서는 Crawling한 305,310개의 Tweet 데이터를 사용하였으며 Unigram/Bigrams/Personal-pronouns/Adjectives/Word-net Affect emotion lexicon/Word-net Affect emotion lexicon with left/right context/Word-net Affect emotion POS/POS/POS-Bigrams/Dependency-Parsing Feature/Emoticons를 feature로 사용하였다.<br>
**모델 구현:**<br>
 multi-class svm 모델을 구현할 수 있는 대표적인 사이킷런의 패키지 3가지(SGDClassifier, LinearSVC, SVC)중 최적의 성능을 내는 알고리즘을 찾고자 하였다. 세 패키지의 알고리즘의 차이는 LinearSVC와 SVC는 각각 멀티클래시피케이션을 ‘one vs rest’와 ‘one vs one’ 방식으로 한다는 점에서 차이가 있었고 SGDClassifier는 liblinear와 libsvm을 사용하는 두 알고리즘과 다르게 stochastic gradient descent 방식으로 옵티마이징하였다.
 각 모델들을 파라미터 튜닝을 해가며 데이터를 7:3 비율로 나누어 train/test set으로 검증 결과에 대해 비교하였다. 그렇게 train/test set에 대한 검증 결과를 가장 높게 나오도록 하는 파라미터를 가지고 정확한 성능 검정을 위해 5-fold 교차 검증으로 세 모델의 성능을 비교하였다. 세 알고리즘의 5-fold의 결과 SGDClassifier가 타 모델에 비해 성능이 0.01이 더 좋았다. 뿐만 아니라 모델의 training 속도도 가장 빨랐다. 그렇게 SGDClassifier 패키지의 모델을 사용하여 감정 분류를 하기로 결정하였다. 이후 SVM의 성능을 높이는 대표적인 방법 중 하나인 Bagging으로 성능을 높여보고자 하였지만 유의미한 결과를 내지는 못하였다.<br>

결국 SGDClassifier로 multi-class svm을 구현하였고, 이후 레이블링이 되어있는 train set을 학습시킨 후 CalibratedClassifierCV를 사용하여 레이블링이 되어있지 않은 200개의 문장이 담긴 test set에 대하여, 각 문장별로 모델이 낸 예측 확률과 클래스 결과를 얻을 수 있도록 하였다.<br>
```python
    # SGDClassifier - loss = 'hinge'
    cvect = CountVectorizer()

    cvect_X_train = cvect.fit(X_train)
    cvect_X_test = cvect.fit(X_test)
    print(cvect.vocabulary_)

    cvect_X_train = cvect.transform(X_train).toarray()
    cvect_X_test = cvect.transform(X_test).toarray()

    tfid = TfidfTransformer()

    tt_X_train = tfid.fit(cvect_X_train)
    tt_X_test = tfid.fit(cvect_X_test)

    tt_X_train = tfid.transform(cvect_X_train).toarray()
    tt_X_test = tfid.transform(cvect_X_test).toarray()

    sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-2, random_state=42, max_iter=10, tol=None)
    sgd = sgd.fit(tt_X_train, y_train)

    y_pred = sgd.predict(tt_X_test)

    clf = CalibratedClassifierCV(sgd)
    clf.fit(tt_X_train, y_train)

    y_proba = clf.predict_proba(tt_X_test)

    print("training accuracy:", sgd.score(X_train, y_train))
    print(clf.classes_)
```

## k-fold
k-fold: k개의 데이터 셋을 만든 후 k번 만큼 학습과 검증을 수행하는 방법<br><br>
overfitting의 위험 없이 accuracy를 확인하기 위해
k-fold를 통해 SGDClassifier 모델을 조금 더 정확하게 평가를 해 보았다.<br><br>
결과:<br>
|K-fold Score|Score|
|:---|:---|
|3-fold score mean|0.697|
|5-fold score mean|0.736|
|10-fold score mean|0.749|
```pyhton
    #3-fold
    scores = cross_val_score(sgd,  df_train.text, df_train.label, cv=3)
    print('cross-val-score \n{}'.format(scores))
    print('cross-val-score.mean \n{:.3f}'.format(scores.mean()))

    #5-fold
    scores = cross_val_score(sgd, df_train.text, df_train.label, cv=5)
    print('cross-val-score \n{}'.format(scores))
    print('cross-val-score.mean \n{:.3f}'.format(scores.mean()))

    #10-fold
    scores = cross_val_score(sgd,  df_train.text, df_train.label, cv=10)
    print('cross-val-score \n{}'.format(scores))
    print('cross-val-score.mean \n{:.3f}'.format(scores.mean()))
```

## Check the predicted percentage/class distribution
nmt api별로 번역한 데이터셋에 대한 예측 확률이 몇 퍼센트대에 주로 머무는지 확인하고자 하였다.
이와 더불어 nmt api들끼리 예측된 클래스의 분포가 비슷한지 파악하고자 하였다.<br>
각 nmt_api로 번역한 test set에 대한 예측 확률과 클래스 분포를 살펴 보았다.

### Google_nmt_api test set
- predicted percentage distribution
    |Accuracy|Count|
    |:---|:---|
    |30%|87|
    |20%|50|
    |40%|49|
    |50%|10|
    |10%|4|

- predicted class distribution
    |Label|Count|
    |:---|:---|
    |neutral|84|
    |disgust|84|
    |sadness|8|
    |happiness|7|
    |anger|6|
    |fear|6|
    |surprise|5|

### Kakao_nmt_api test set
- predicted percentage distribution
    |Accuracy|Count|
    |:---|:---|
    |30%|72|
    |20%|62|
    |40%|45|
    |50%|13|
    |10%|7|
    |60%|1|

- predicted class distribution
    |Label|Count|
    |:---|:---|
    |disgust|85|
    |neutral|79|
    |happiness|11|
    |fear|8|
    |sadness|8|
    |surprise|6|
    |anger|3|

### Papago_nmt_api test set
- predicted percentage distribution
    |Accuracy|Count|
    |:---|:---|
    |30%|78|
    |40%|49|
    |20%|48|
    |50%|12|
    |10%|2|

- predicted class distribution
    |Label|Count|
    |:---|:---|
    |disgust|94|
    |neutral|68|
    |happiness|8|
    |sadness|8|
    |anger|4|
    |fear|4|
    |surprise|3|

```python
    X_test = list(X_test)
    print(X_test)

    #확률 분포 파악
    i = 0
    look = []
    look_class = []
    for i in range(len(X_test)):
      score = np.max(y_proba[i,:])*100

      idx = np.argmax(y_proba[i,:])
      emotion = clf.classes_[idx]

      if score>=0 and score <10 :
          print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))
          look.append('0%')
          look_class.append(emotion)
      elif score >= 10 and score <20:
          print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))
          look.append('10%')
          look_class.append(emotion)
      elif score >= 20 and score <30:
          print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))
          look.append('20%')
          look_class.append(emotion)
      elif score >= 30 and score <40:
          print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))
          look.append('30%')
          look_class.append(emotion) 
      elif score >= 40 and score <50:
          print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))
          look.append('40%')
          look_class.append(emotion) 
      elif score >= 50 and score <60:
          print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))
          look.append('50%')
          look_class.append(emotion)   
      elif score >= 60 and score <70:
          print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))
          look.append('60%')
          look_class.append(emotion)
      elif score >= 70 and score <80:
          print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))   
          look.append('70%')
          look_class.append(emotion)
      elif score >= 80 and score <90:
          print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))   
          look.append('80%')
          look_class.append(emotion)
      elif score >= 90 and score <100:
          print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))
          look.append('90%')
          look_class.append(emotion)
      else:
          print("[{}]는 {:.2f}% 확률로 {} 리뷰이지 않을까 추측해봅니다.\n".format(X_test[i], score, emotion))  
          look.append('100%')
          look_class.append(emotion)

    #각 nmt api로 번역한 test set 예측 확률 / 분포 확인
    #Google
    look_google =pd.DataFrame(look)
    print("predicted percentage distribution:\n", look_google[0].value_counts())
    look_class_google = pd.DataFrame(look_class)
    print("\npredicted class distribution:\n", look_class_google[0].value_counts())

    #Kakao
    look_kakao =pd.DataFrame(look)
    print("predicted percentage distribution:\n", look_kakao[0].value_counts())
    look_class_kakao = pd.DataFrame(look_class)
    print("\npredicted class distribution:\n", look_class_kakao[0].value_counts())

    #Papago
    look_papago =pd.DataFrame(look)
    print("predicted percentage distribution:\n", look_papago[0].value_counts())
    look_class_papago = pd.DataFrame(look_class)
    print("\npredicted class distribution:\n", look_class_papago[0].value_counts())
```

## 결과 저장
```python
    #결과 저장
    df_X_test = pd.DataFrame({'text':X_test})
    df_look = pd.DataFrame({'prob':look})
    df_look_class = pd.DataFrame({'class':look_class})

    df_result = pd.concat([df_X_test, df_look_class, df_look], axis = 1)
    print(df_result)

    df_result.to_csv('/content/drive/My Drive/google_result_svm.csv')
```

## Trials
아래는 성능 개선과 모델 별 성능 비교를 위해 사용한 코드이다.
|Package|Parameter|Best Accuracy|5-fold Acuuracy(mean)|
|:---|:---|:---|:---|
|SGDClassifier|penalty, alpha, random_state, max_iter|0.747|0.736|
|LinearSVC|C|0.741|0.735|
|SVC|kernel, gamma, C|0.748|0.748|
|SDGClassifier + Bagging|n_estimators, max_samples|0.745|   -   |

```python
    #Bagging(SDGClassifier)
    from sklearn.ensemble import BaggingClassifier

    param = [{'loss':['hinge'], 'penalty':['l2'],'alpha':[1e-2], 'random_state':[42], 'max_iter':[10], 'tol':[None]}]
    sgd_grid = GridSearchCV(SGDClassifier(),param_grid=param, scoring='accuracy')

    n_estimators = 100
    n_jobs = 5
    model = BaggingClassifier(base_estimator=sgd_grid,
                              n_estimators=n_estimators,
                              max_samples=0.9,
                              n_jobs=n_jobs)

    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', model),
                   ])

    sgd.fit(X_train, y_train)

    y_pred = sgd.predict(X_test)

    print("training accuracy:", sgd.fit(X_train, y_train).score(X_train, y_train))
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=my_tags))

    #Linear SVC
    lsvc = LinearSVC(C=100, class_weight=None, dual=True, fit_intercept=True)

    lsvc = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', lsvc),])
    lsvc.fit(X_train, y_train)

    y_pred = lsvc.predict(X_test)

    print("train accuracy:", lsvc.fit(X_train, y_train).score(X_train, y_train))
    print('test accuracy: %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=my_tags))

    #SVC
    #load data
    df = pd.read_csv('/content/drive/Shared drives/데이터분석캡스톤디자인/데이터/라벨 데이터/train.txt',sep=';', names =['text','label'])

    X = df.text
    y = df.label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3], 'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    svc_grid = GridSearchCV(SVC(kernel = 'linear',C = 1000), params_grid, scoring='accuracy',cv=10)
    svc = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', svc_grid ),
                   ])

    svc = svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    print('Best score for training data:', svc_grid.best_score_,"\n")
    print('Best C:',svc_grid.best_estimator_.C,"\n")
    print('Best Kernel:',svc_grid.best_estimator_.kernel,"\n")
    print('Best Gamma:',svc_grid.best_estimator_.gamma,"\n")
```
