#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('wordnet', quiet=True)
#%%
df = pd.read_csv('Dataset/reviews.txt')
df.head()
# %%
df.isnull().sum()
# %%
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word isalpha()
              and word not in stop_words]
    return " ".join(tokens)
#%%
df['processed_comments'] = df['cleaned_comments'].apply(preprocess_text)
tokenizer = Tokenizer(num_words = 500, oov_tokens ="<OOV>")
tokenizer.fit_on_texts(df['processed_comments'])
sequences = tokenizer.texts_to_sequences(df['processed_comments'])

#%%
max_length = max(len(x) for x in sequences)
padded_sequences = pad_sequences(sequences,maxlen = max_length, padding = 'post')
#%%
labels = pd.get_dummies(df['sentiment']).values

#%%
X_train, X_test,y_train,y_test = train_test_split(padded_sequences,labels,test_size=0.2,random_state=42)
#%%
model = Sequential()
model.add(Embedding(5000,128,input_length=max_length))
model.add(LSTM(64,dropout=0.2,recurrent_dropoout=0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(labels.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
#%%
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
history = model.fit(X_train, y_train, epochs=100,batch_size=128,validation_split=0.2,callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test,y_test)
print(f"test Accuracy:{accuracy * 100}%")