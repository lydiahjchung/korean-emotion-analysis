from googletrans import Translator

#Data 불러오기
file = open('/content/drive/Shared drives/데이터분석캡스톤디자인/데이터/최종 test 데이터/twitter_100.txt',"r")
data_1 = file.readlines()
data_train = []
for i in data_1:
  data_train.append(i[:-1])
print(data_train)

for i in range(len(data_train)):
  string = data_train[i]
  translator = Translator(proxies=None, timeout=None)
  result = translator.translate(string, dest="en")
  print(result.text)

#Test Code
translator = Translator()
result = translator.translate('안녕하세요.', dest="en")
print(result.text)