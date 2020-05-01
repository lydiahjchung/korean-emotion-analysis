from twitterscraper.query import query_tweets
from twitterscraper.tweet import Tweet

import datetime
import time
import json

# Twitter 크롤링
start = time.time()  # 시작 시간 저장
list_of_tweets = query_tweets('총선', begindate=datetime.date(2020,4,1), 
                                enddate=datetime.date(2020,4,30))

# JSON 저장
file_path = "./data_0401_0430.json"
data = []
count = 1

for tweet in list_of_tweets:
    data.append({
        "id": count,
        "content": tweet.text
    })
    count += 1
    

# 파일 저장
with open(file_path, "w", encoding="utf-8") as make_file:
    json.dump(data, make_file, indent="\t", ensure_ascii = False)

total_time = time.time() - start
hours = total_time // 3600
minutes = (total_time - (hours * 3600)) // 60
seconds = total_time - (hours * 3600) - (minutes * 60)  
print("{0}시간 {1}분 {2}초가 소요되었습니다.".format(hours, minutes, seconds))