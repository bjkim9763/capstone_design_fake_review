import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split as tts
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
# 한글 stopwords 설정 구글에서 다운로드
file_path = "stopwords-ko.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    stopwords = f.readlines()

stopwords = [stopword.rstrip('\n') for stopword in stopwords]
# 자연어 처리 함수
def clean_text(text):
    return ''.join([re.sub('[^ A-Za-z0-9가-힣\s]', '', i) for i in text])
    #자연어처리

# 데이터 로드하기
df = pd.read_csv('total_dataset.CSV')
df['body'] = df['body'].astype(str).apply(clean_text) # 전처리
df['body'] = df['body'].replace('^ +',"")
df['body'].replace('',np.nan, inplace=True)   #공백제거
# print(df.isnull().sum())
df = df.dropna(how = 'any') #
# print(df.isnull().sum())

# 토큰화
body = df['body']
label = df['label']
x_train, x_valid, y_train, y_valid = tts(body, label, test_size=0.2, shuffle=True, stratify=label, random_state=34)

okt = Okt()
'''
X_train = []
for sentence in tqdm(x_train):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(stopwords_removed_sentence)'''
'''
X_test = []
for sentence in tqdm(x_valid):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_test.append(stopwords_removed_sentence)
'''
# 정수 인코딩
'''with open('X_train.txt','wb') as f:
    pickle.dump(X_train, f)'''
with open('X_train.txt','rb') as f:
    X_train = pickle.load(f)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
'''print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
'''
vocab_size = total_cnt - rare_cnt + 1 # 단어 집합의 크기

tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_train)
#X_train = tokenizer.texts_to_sequences(X_train)
#X_test = tokenizer.texts_to_sequences(X_test)

# print(X_train[:3])

#y_train = np.array(y_train)
#y_test = np.array(y_valid)

'''print('리뷰의 최대 길이 :',max(len(review) for review in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))'''


'''plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()'''

'''def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))
'''
max_len = 900
'''below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)'''
'''
embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
'''
from selenium import webdriver
import time

options = webdriver.ChromeOptions() # 크롬 옵션 객체 생성
options.add_argument('headless') # headless 모드 설정
options.add_argument("window-size=1920x1080") # 화면크기(전체화면)
options.add_argument("disable-gpu")
options.add_argument("disable-infobars")
options.add_argument("--disable-extensions")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko")
prefs = {'profile.default_content_setting_values': {'cookies' : 2, 'images': 2, 'plugins' : 2, 'popups': 2, 'geolocation': 2,
                                                    'notifications' : 2, 'auto_select_certificate': 2, 'fullscreen' : 2, 'mouselock' : 2,
                                                    'mixed_script': 2, 'media_stream' : 2, 'media_stream_mic' : 2, 'media_stream_camera': 2,
                                                    'protocol_handlers' : 2, 'ppapi_broker' : 2, 'automatic_downloads': 2, 'midi_sysex' : 2,
                                                    'push_messaging' : 2, 'ssl_cert_decisions': 2, 'metro_switch_to_desktop' : 2, 'protected_media_identifier': 2,
                                                    'app_banner': 2, 'site_engagement' : 2, 'durable_storage' : 2}}
options.add_experimental_option('prefs', prefs)

browser = webdriver.Chrome('./chromedriver.exe', options=options)
file_path = "stopwords-ko.txt"
'''
with open(file_path, 'r', encoding='utf-8') as f:
  stopwords = f.readlines()

stopwords = [stopword.rstrip('\n') for stopword in stopwords]
'''
loaded_model = load_model('best_model.h5')

def get_blog_text(url):
    browser.get(url)
    time.sleep(1)
    try:
        pop_up = browser.find_element_by_css_selector('iframe#mainFrame')
        browser.switch_to.frame(pop_up)
        text = browser.find_element_by_css_selector('body').text
    except:
        text = browser.find_element_by_css_selector('body').text
    return text.replace('\n',' ')

def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{:.2f}% 확률로 진짜 리뷰입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 가짜 리뷰입니다.\n".format((1 - score) * 100))
while True:
    url = input("url을 입력하세요: \n")
    new_sentence = get_blog_text(url)
    sentiment_predict(new_sentence)
