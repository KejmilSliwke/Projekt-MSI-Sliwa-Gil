import numpy as np
from sklearn import  neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 170)



##################################  UZYWAMY KLASYFIKATORA KNN Z SCIKIT LEARN    ############################################################

'''
#pobieramy dane z pliku tekstowego
df = pd.read_csv('breast-cancer-wisconsin.data')

#Zamieniamy 16 brakujacych wartosci na -99999, poniewaz wartosci atrybutow musza byc liczba! -99999 wiekszosc algorytmow traktuje jako outlayer? cos takiego. A jesli
#mielibysmy za kazdym razem usuwac cale kolumny, w ktorych nie mamy paru wartosci (co w rzeczywistosci zdarza sie bardzo czesto), to musielibymy utracic mnostwo danych. Co jest
#bez sensu.

df.replace('?', -99999, inplace=True)
#usuwamy kolumne id, poniewaz nie wplywa ona na zlosliwosc guza
df.drop(['id'], 1, inplace=True)


#bierzemy atrybuty wyrzucajac tylko ostatnia kolumne, ktora jest etykieta
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

#Podzial zbioru na czesc treningowa i testowa
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#Przykladowy zbior treningowy wraz z etykietami
print('Przykladowe atrybuty wraz z etykieta:')
for x in range(0,2):
    print(X_train[x], y_train[x])


#Definiujemy uzywany klasyfikator, jakim jest w tym przypadku KNN
KNN = neighbors.KNeighborsClassifier()
#uczymy klasyfikator na podstawie zbioru treningowego. Ja rozumiem to w ten sposob, ze klasyfikator przyznaje wagi poszczegolnym atrybutom,
#bo w przeciwnym razie po co go w ogole trenowac, skoro i tak bierze tylko odleglosci od sasiadow? A tak to moze atrybuty maja rozna wage.
KNN.fit(X_train, y_train)

#sprawdzamy nasza dokladnosc zbiorem testowym
accuracy = KNN.score(X_test, y_test)
print('Dokladnosc(accuracy) = % .3f' % accuracy)


#[[]] - list of lists
#tutaj sobie wymyslamy jakies losowe atrybuty
example_measures = np.array([[4,8,7,5,5,6,7,2,1],[4,2,1,2,2,2,3,2,1]])
#dostosowanie wymiarow do oryginalnej tablicy
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = KNN.predict(example_measures)
prediction = prediction.reshape(2, 1)
print('Wymyslone atrybuty: \n', example_measures,'\nEtykiety wymyslonych atrybutow: \n',prediction)
'''


############################################### TERAZ PISZEMY SWOJ KNN  ##############################################################




#KNN dziala na zasadzie odleglosci euklidesowej, mozna opisac wzor ze to
#PIERWIASTEK(SUMA;i=1;n(qi-pi)^2), n to liczba wymiarow
####################### Podstawy odleglosci euklidesowej
'''
from math import sqrt
punkt1 = [1,3]
punkt2 = [2,5]
euclidian_distance = sqrt( (punkt1[0]-punkt2[0])**2 + (punkt1[1]-punkt2[1])**2 )
print(euclidian_distance)
'''



########################## Dalsza czesc wlasnego KNN


import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

#uzywajac {} tworzymy slownik (dictionary), klase "k" oraz klase 'r'
dataset = { 'k': [ [1,2], [2,3], [3,1] ], 'r':[ [6,5], [7,7], [8,6] ] }
#pojawia sie nowy punkt, bez przyporzadkowanej klasy
new_features = [5,7]

'''
#nanoszenie kazdego punktu z datasetu na plaszczyzne
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1], s=100, color=i)
#nowy punkt na plaszyznie
plt.scatter(new_features[0], new_features[1])
plt.show()
'''

def k_nearest_neighbours(data, predict, k=3)
    if len(data) >= k:
        warnings.warn('K jest ustawione na wartosc mniejsza niz ilosc wszystkich klas!')

    knnalgos
    return vote_result