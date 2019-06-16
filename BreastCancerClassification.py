'''
import numpy as np
from sklearn import  neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 170)



##################################  UZYWAMY KLASYFIKATORA KNN Z SCIKIT LEARN    ############################################################


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
print('Przykadowe atrybuty ze zbioru treningowego wraz z ich etykieta:')
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
print('Losowe atrybuty (nie bedace w zadnym z poprzednich zbiorow treningowych i testowych): \n', example_measures,'\nEtykiety dla tych atrybutow atrybutow: \n',prediction)
'''


############################################### TERAZ PISZEMY SWOJ KNN  ##############################################################




#KNN dziala na zasadzie odleglosci euklidesowej, mozna opisac wzor ze to
#PIERWIASTEK(SUMA;i=1;n(qi-pi)^2), n to liczba wymiarow
####################### Podstawy odleglosci euklidesowej
'''
from math import sqrt
p1 = [1,3]
p2 = [2,5]
euclidean_distance = sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
print('Koordynaty punktu pierwszego: ', p1[0],p1[1], '\nKoordynaty punktu drugiego: ' ,p2[0],p2[1],
'\nOdleglosc miedzy punktami: % .2f' % euclidean_distance)
'''



########################## Dalsza czesc wlasnego KNN

'''
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
new_features = [4,4]


#nanoszenie kazdego punktu z datasetu na plaszczyzne
#for i in dataset:
    #for ii in dataset[i]:
        #plt.scatter(ii[0],ii[1], s=100, color=i)
#nowy punkt na plaszyznie
#plt.scatter(new_features[0], new_features[1])
#plt.show()


def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K jest ustawione na wartosc mniejsza niz ilosc wszystkich klas!')

    distances = []
    for group in data:
        for features in data[group]:
            #szybsza wersja obliczen odleglosci dzieki algorytmom z biblioteki
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    #jak juz posortujemy odleglosci chcemy tylko te do k
    # for i in sorted(distances)[:k]:
    #    i[i]
    # to to samo co
    votes = [i[1] for i in sorted(distances)[:k]]

    #wyswietla jaka grupa byla najczesciej glosowana, oraz ile bylo na nia glosow
    print(Counter(votes).most_common(1))

    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

#wyswietla koncowy wynik glosowania
result = k_nearest_neighbours(dataset, new_features, k=3)

print(result)


#rysowanie punktow na plaszczyznie
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1], s=100, color=i)
#nowy punkt na plaszyznie
plt.scatter(new_features[0], new_features[1], color = result)
plt.show()
'''

###################################     POROWNUJEMY NASZ DO TEGO Z SCIKIT ################
'''
import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 170)



def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K jest ustawione na wartosc mniejsza niz ilosc wszystkich klas!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

full_data = df.astype(float).values.tolist()
print('Pierwsze 3 rekordy z pliku csv przed przemieszaniem:','\n',full_data[0],
      '\n',full_data[1],'\n',full_data[2])
random.shuffle(full_data)
print('\nPierwsze 3 rekordy z pliku csv po przemieszaniu','\n',full_data[0],
      '\n',full_data[1],'\n',full_data[2])

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbours(train_set, data, k=5)
        #Jesli grupa przewidziana przez nasz KNN jest zgodna z grupa w zbiorze
        #
        if group == vote:
            correct +=1
        total +=1
print('\nDokladnosc naszego klasyfikatora KNN: % .3f' % float(correct/total))
'''


#########################TERAZ BEDZIEMY POROWNYWAC Z TYM Z SCIKIT LEARN I PEWNOSC#######


#jesli bedziemy zwiekszac nasze k to accuracy bedzie stopniowo malec

import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 170)


def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K jest ustawione na wartosc mniejsza niz ilosc wszystkich klas!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    #print(vote_result, confidence)
    return vote_result, confidence

proby = 300
accuracies= []
for i in range(proby):
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?',-99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    #print('Pewnosc wyboru klas, ktore zostaly blednie ocenione w procesie predykcji:')
    correct = 0
    total = 0
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbours(train_set, data, k=5)
            if group == vote:
                correct +=1
            #else:
                #print(confidence)
            total +=1
    #print('Dokladnosc naszego klasyfikatora: % .3f' % float(correct/total))
    accuracies.append((correct/total))

print('Srednia dokladnosc naszego klasyfikatora KNN przy',proby,'probach:',sum(accuracies)/len(accuracies),'\n')



from sklearn import  neighbors
from sklearn.model_selection import train_test_split
accuracies= []

for i in range(proby):
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    X = np.array(df.drop(['class'],1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


    KNN = neighbors.KNeighborsClassifier()
    KNN.fit(X_train, y_train)
    accuracy = KNN.score(X_test, y_test)
    #print('Dokladnosc klasyfikatora sci-kit learn = % .3f' % accuracy)
    accuracies.append(accuracy)

print('Srednia dokladnosc KNN sci-kit learn przy',proby,'probach:',sum(accuracies)/len(accuracies))
