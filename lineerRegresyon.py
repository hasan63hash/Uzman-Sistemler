

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# veriyi dosyadan oku
dataset=pd.read_csv('Salary_Data.csv')

type(dataset)

#dataset.head()

X=dataset.iloc[:,:-1].values

y=dataset.iloc[:,-1].values

#Veri setini böl

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_train,y_train)

#test verisini tahmin etmeye calis
# y_tahmin:modelin tahmini y_test:veri setindeki degerler
y_tahmin=regressor.predict(X_test)

# Grafik çizme
# Eğitim sonuçları
plt.scatter(X_train,y_train,color='red')

plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title("Maas vs. Deneyim (Eğitim Seti)")

plt.xlabel('Deneyim')
plt.ylabel('Maas')
plt.show()

#Test sonuçları
plt.scatter(X_test,y_test,color='red')

plt.plot(X_test,regressor.predict(X_test), color='blue')
plt.title("Maas vs. Deneyim (Eğitim Seti)")

plt.xlabel('Deneyim')
plt.ylabel('Maas')
plt.show()

#Tahmin üretme
print(regressor.predict([[12]]))

#Coef and intercept
print(regressor.coef_)
print(regressor.intercept_)

