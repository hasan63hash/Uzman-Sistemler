
import pandas as pd

dataset=pd.read_csv('veriseti4.csv')

# 3. indexe sahip kolondan başlar son kolona kadar (son kolon dahil değil) olan verileri alır
X=dataset.iloc[:,3:13].values

y=dataset.iloc[:,13].values

# y=dataset.iloc[:,-1].values  bu şekilde son kolonu alır.

# veri setimizdeki eksik verilerin kolon bazında toplam sayısını verir.
dataset.isnull().sum()

#Sayısal Veriye Çevirme
from sklearn.preprocessing import LabelEncoder

labelencoder_X1=LabelEncoder()

X[:,1]=labelencoder_X1.fit_transform(X[:,1])

labelencoder_X2=LabelEncoder()

X[:,2]=labelencoder_X2.fit_transform(X[:,2])


#Veri setimizi eğitim ve test olarak bölelim

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=26)

#Özellik Ölçeklendirme

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#Yapay Sinir Ağını Oluşturmaya Başla

import keras
from keras.models import Sequential
from keras.layers import Dense

siniflandirici=Sequential()
#1. gizli katman
siniflandirici.add(Dense(input_dim=10,units=5,activation='relu', kernel_initializer='uniform'))

#2. gizli katman
siniflandirici.add(Dense(units=7,activation='relu', kernel_initializer='uniform'))

#Çıktı katmanı
siniflandirici.add(Dense(activation='sigmoid',units=1,kernel_initializer='uniform'))

#Derleme
siniflandirici.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'],)

siniflandirici.fit(X_train,y_train,batch_size=10,epochs=10)

y_tahmin=siniflandirici.predict(X_test)



        


