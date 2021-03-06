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


#trainset 불러오기
df_train = pd.read_csv('/content/drive/Shared drives/데이터분석캡스톤디자인/데이터/라벨 데이터/labeled_final.txt',sep=';', names =['text','label'])
X_train = df_train.text
y_train = df_train.label
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

#testset 최종 전처리
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

#레이블 분포 확인
df_train['label'].value_counts()

#레이블 분포 시각화
my_tags = ['anger','happiness','surprise', 'sadness', 'fear', 'neutral', 'disgust']
plt.figure(figsize=(10,4))
y_train.value_counts().plot(kind='bar');

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
df_test = pd.read_csv('/content/drive/Shared drives/데이터분석캡스톤디자인/데이터/최종 test 데이터/papago_final_test.txt',sep=';', names =['text','label'])
X_test = df_test.text
y_test = df_test.label

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

#k-fold
#3-fold
scores = cross_val_score(sgd,  df.text, df.label, cv=3)
print('cross-val-score \n{}'.format(scores))
print('cross-val-score.mean \n{:.3f}'.format(scores.mean()))

#5-fold
scores = cross_val_score(sgd, df.text, df.label, cv=5)
print('cross-val-score \n{}'.format(scores))
print('cross-val-score.mean \n{:.3f}'.format(scores.mean()))

#10-fold
scores = cross_val_score(sgd,  df.text, df.label, cv=10)
print('cross-val-score \n{}'.format(scores))
print('cross-val-score.mean \n{:.3f}'.format(scores.mean()))

#전체 test set에 대한 예측 확률/분포 결과
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

#결과 저장
df_X_test = pd.DataFrame({'text':X_test})
df_look = pd.DataFrame({'prob':look})
df_look_class = pd.DataFrame({'class':look_class})

df_result = pd.concat([df_X_test, df_look_class, df_look], axis = 1)
print(df_result)

df_result.to_csv('/content/drive/My Drive/google_result_svm.csv')

#---Trials---#
#Bagging(SDGClassifier)
from sklearn.ensemble import BaggingClassifier

param = [{'loss':['hinge'], 'penalty':['l2'],'alpha':[1e-2], 'random_state':[42], 'max_iter':[10], 'tol':[None]}]
sgd_grid = GridSearchCV(SGDClassifier(),param_grid=param, scoring='accuracy')

n_estimators = 1000
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
