try:
  from soyspacing.countbase import CountSpace
  from soyspacing.countbase import RuleDict
except:
  #!pip install soyspacing
  from soyspacing.countbase import CountSpace
  from soyspacing.countbase import RuleDict

import re
import json

# Soyspacing 모델 학습하기
corpus_fname = './134963_norm.txt'
rule_dict = RuleDict('./space_rules.txt')
model = CountSpace()
model.train(corpus_fname)

# Soyspacing parameter 정하기
verbose=False
mc = 10  # min_count
ft = 0.3 # force_abs_threshold
nt =-0.3 # nonspace_threshold
st = 0.3 # space_threshold

for i in range(1, 30, 2):
  if i < 9:
    i_start = "0{}".format(i)
    i_end = "0{}".format(i + 1)
  elif i == 9:
    i_start = "09"
    i_end = "10"
  else:
    i_start = i
    i_end = i + 1

  # 전처리할 JSON 파일 열기
  with open('data_04{}_04{}.json'.format(i_start, i_end), 'r') as f:
    json_data = json.load(f)

  temp = '@@@@@@@@@@@@@@@@@@@@@@'
  with open('pre_data_04{}_04{}.txt'.format(i_start, i_end), "w", encoding="utf-8") as f:

    # 전처리하기
    for sentence in json_data:

      remove_hypterlink = re.sub(r"http\S+", "", sentence['content'])       # 하이퍼링크 제거
      remove_twitterlink = re.sub(r"pic\S+", "", remove_hypterlink)         # 트위터링크 제거
      remove_retweet = re.sub(r"@\S+", "", remove_twitterlink)              # 트위터아이디 제거

      # 불용어 처리
      sub_sent_1 = remove_retweet.replace('\n', '').replace('=','').replace('#', '').replace('KBS', '').replace('ㅎ', '').replace('ㄴ', '').replace('^', '')
      sub_sent_2 = sub_sent_1.replace('…', '').replace('ㅋ', '').replace('ㅠ', '').replace('다음뉴스', '').replace('SBS', '').replace('ㄱ', '').replace('오마이뉴스','')
      sub_sent_3 = sub_sent_2.replace('다음 뉴스', '').replace('기자', '').replace('<', '').replace('> ', '').replace('｜', '').replace('연합뉴스', '').replace('출처', '')
      sub_sent_4 = sub_sent_3.replace('/', '').replace('| ', '').replace('YTN', '').replace("인천투데이", '').replace('》', '').replace('《', '').replace('〈', '').replace('〉', '')
      sub_sent_5 = sub_sent_4.replace('sbs', '').replace('kbs', '').replace('ㅉ', '').replace('아시아경제', '').replace('●', '').replace('*', '').replace('모바일', '')
      sub_sent_6 = sub_sent_5.replace('■', '').replace('.', '').replace('·', '').replace('?','').replace('!', '').replace('~', '').replace('+', '')
      sub_sent_7 = sub_sent_6.replace('경향신문', '').replace('중앙일보', '').replace(':', '').replace('네이버','').replace('ㅜ', '').replace('~', '').replace('님이 공유', '')
      sub_sent_8 = sub_sent_7.replace('ㅂ', '').replace('ㅅ', '').replace('-', '').replace('_','').replace('뉴데일리', '').replace('ㅡ', '').replace('ㅈㅇ당', '정의당')
      sub_sent_9 = sub_sent_8.replace('a', '').replace('b', '').replace('c', '').replace('d', '').replace('e','').replace('f', '').replace('g', '').replace('h', '').replace('i','').replace('j', '').replace('k', '').replace('l', '').replace('m','').replace('n', '').replace('o', '').replace('p', '').replace('q','').replace('r', '').replace('s', '').replace('t','').replace('u', '').replace('v', '').replace('w', '').replace('x','').replace('y','').replace('z','')
      sub_sent_10 = sub_sent_9.replace('A', '').replace('B', '').replace('C', '').replace('D', '').replace('E','').replace('F', '').replace('G', '').replace('H', '').replace('I','').replace('J', '').replace('K', '').replace('L', '').replace('M','').replace('N', '').replace('O', '').replace('P', '').replace('Q','').replace('R', '').replace('S', '').replace('T','').replace('U', '').replace('V', '').replace('W', '').replace('X','').replace('Y','').replace('Z','')
      sent = sub_sent_10.strip()

      # 중복 문장 제거
      if temp not in sent:
        temp = sent

        list_sent = list(sent)

        # [] 내용 제거
        if '[' in list_sent:
          start = list_sent.index('[')
          if ']' in list_sent:
            end = list_sent.index(']')
            del list_sent[start : end + 2]
            sent = ''.join(list_sent)
          else:
            del list_sent[start:]
            sent = ''.join(list_sent)
        
        # () 내용 제거
        if '(' in list_sent:
          start = list_sent.index('(')
          if ')' in list_sent:
            end = list_sent.index(')')
            del list_sent[start : end + 2]
            sent = ''.join(list_sent)
          else:
            del list_sent[start:]
            sent = ''.join(list_sent)

        # 띄어쓰기 모델 parameter
        sent_corrected, tags = model.correct(
            doc=sent,
            verbose=verbose,
            force_abs_threshold=ft,
            nonspace_threshold=nt,
            space_threshold=st,
            min_count=mc,
            rules=rule_dict)
        
        # TXT 파일 저장
        f.write("{}\n".format(sent_corrected))