#Kutuphaneleri yukle
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Egitim veri seti yukle 
egitim_veriseti= pd.read_csv('iris_training.csv')

egitim_x=egitim_veriseti.iloc[:,0:4].values
egitim_y=egitim_veriseti.iloc[:,4].values

#Encodig uygula- egitim_y verisine
encoding_egitim_y=np_utils.to_categorical(egitim_y)

# Test veri seti yukle 
test_veriseti= pd.read_csv('iris_test.csv')

test_x=test_veriseti.iloc[:,0:4].values
test_y=test_veriseti.iloc[:,4].values

#Encodig uygula- test_y verisine
encoding_test_y=np_utils.to_categorical(test_y)

#Yapay Sinir ağını oluştur

model=Sequential()
#1. gizli katmanı ekle
model.add(Dense(4,input_dim=4,activation='relu'))
#2. gizli katman
model.add(Dense(4, activation='relu'))

#çıktı katmanı
model.add(Dense(3,activation='softmax'))
# Modeli calıştır
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

#SECENEK 1
#yapay sinir ağını eğit
model.fit(egitim_x,encoding_egitim_y, epochs=300, batch_size=10)

#yapay sinir ağını test et
sonuclar=model.evaluate(test_x,encoding_test_y)
print("Accuracy: %.2f%%" %(sonuclar[1]*100))

#SECENEK 2
#Yapay sinir ağını hem test edip hem eğitmek istesek
histories3=model.fit(egitim_x,encoding_egitim_y, epochs=300,batch_size=10, validation_data=(test_x, encoding_test_y))

plt.subplot(211)
plt.title('Cross Entropy Loss')
plt.plot(histories3.history['loss'], color='blue', label='train')
plt.plot(histories3.history['val_loss'], color='orange', label='test')
plt.show()

plt.subplot(212)
plt.title('Classification Accuracy')
plt.plot(histories3.history['accuracy'], color='blue', label='train')
plt.plot(histories3.history['val_accuracy'], color='orange', label='test')
plt.show()
