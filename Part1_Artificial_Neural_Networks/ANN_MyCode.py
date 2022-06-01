"""
Theano :
# Calcul numérique basé sur numpy (utilise cpu+gpu)

Tensorflow :
# Calcul numérique (utilisé en recherche en DL) 

Keras :
# Regroupe Theano et Tensorflow, permet de recréer des NN profonds en qql lignes

TIPS : CTRL + i sur une fonction pour afficher l'aide
"""

# ---- ARTIFICIAL NEURAL NETWORK ---- # 

# ---------------------------------------------------------------------------------------------------------------#
# Partie 1 : Préparation des données

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # Colonne 3 à 12 -> variables indépedantes
y = dataset.iloc[:, 13].values   # Colonne 13     -> variable dépendante

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1]) # Geography (country)
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2]) # Gender

# One Hot Encoding GEOGRAPHY
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
# il faut enlever une colonne (donc 0, 0 = forcément l'autre pays)
X = X[:, 1:] 

DATASIZE = len(dataset)
TESTSIZE = 0.2

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TESTSIZE, random_state = 0)

# Standardisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.fit_transform(X_test)

# ---------------------------------------------------------------------------------------------------------------#
# Partie 2 : Construire le réseau de neurones

# Importation des modules de keras
import keras
from keras.models import Sequential # Initialiser le NN
from keras.layers import Dense      # Créer les couches du réseau, s'occupe des poids

# Initialisation
classifier = Sequential()

# Ajout de la couche d'entrée et une couche cachée
classifier.add(Dense(units=6,                           # Couche cachée avec 6 neurones ((11+1)/2)
                     activation="relu",                 # Fonction d'activation
                     kernel_initializer="uniform",
                     input_dim=11))                     # Taille de la couche d'entrée (11 neurones)
                                                        # (nb de variables d'entrée dans le jeu de données)
# Ajout d'une deuxième couche cachée
classifier.add(Dense(units=6,
                     activation="relu",                 
                     kernel_initializer="uniform"))     # Pas besoin de re-préciser la taille de la couche d'entrée

# Ajout de la couche de sortie     
classifier.add(Dense(units=1,                           # 1 neurone de sortie car 1 variable à prédire (quitte ou pas la banque)
                     activation="sigmoid",              # Variable à 2 catégories (= 1 ou 0), si + alors utiliser "softmax"   
                     kernel_initializer="uniform"))

# Compiler le réseau de neurones
classifier.compile(optimizer="adam",                    # adam = algo du gradient stochastique
                   loss="binary_crossentropy",          # Fonction de coût, si + que 2 catégories = categorical_crossentropy
                   metrics=["accuracy"])                # Mesure la perf du modèle

# Entraîner le réseau de neurones
classifier.fit(X_train, y_train,
               batch_size=10,                           # Lot de 10 observations avant rétropropagation
               epochs=100)                              # Les données sont passées 100x dans le NN

# Prédiction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)                                 # Classifier avec un seuil de 50% pour VRAI / FAUX

# Prédiction d'une seule observation
"""
Utiliser le réseau de neurones pour prédire si le client suivant va ou non quitter la banque dans les 6 mois :

Pays : France
Score de crédit : 600
Genre : Masculin
Âge : 40 ans
Durée depuis entrée dans la banque : 3 ans
Balance : 60000 €
Nombre de produits : 2
Carte de crédit ? Oui
Membre actif ? : Oui
Salaire estimé : 50000 €

Devrait-on dire au revoir à ce client ?

"""

new_prediction = classifier.predict(sc.fit_transform(np.array([[0, 0, 600, 0, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)    
"""
--> NON ! Ce client reste.
"""

# Matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

precision = (cm[0, 0] + cm[1, 1]) / (DATASIZE*TESTSIZE)
print("Précision : ", precision)

# K-Fold CrossValidation
from keras.wrappers.scikit_learn import KerasClassifier # Pont entre keras et sklearn
from sklearn.model_selection import cross_val_score
# Améliorer l'ANN
from keras.layers import Dropout # éviter l'overfitting

def build_classifier():
    classifier = Sequential()
    
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))   
    classifier.add(Dropout(rate=0.1)) # 10% de chance qu'un neurone soit désactivé, augmenter la valeur si tj overfit
                 
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))     
    classifier.add(Dropout(rate=0.1))
    
    classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
    
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])   

    return classifier             

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)                              

precision = cross_val_score(estimator=classifier, 
                            X=X_train, 
                            y=y_train, 
                            cv=10)                          # 10 groupes pour la CV

moyenne = precision.mean()*100
ecart_type = precision.std()*100
print("Précision (moyenne) : ", "{:.2f}".format(moyenne), "%")
print("Ecart-type : ", "{:.2f}".format(ecart_type), "%")

""" 
Avant optimisation :
Précision (moyenne) :  84.17 %
Ecart-type :  1.61 %
"""

# OPTIMISATION 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))                    
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))     
    classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
    classifier.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])   

    return classifier             

classifier = KerasClassifier(build_fn=build_classifier)                              
parameters = {"batch_size": [25, 32],
              "epochs" : [100, 500],
              "optimizer" : ["adam", "rmsprop"]}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_precision = grid_search.best_precision_

print("Meilleure précision : ", "{:.2f}".format(best_precision), "%")
print("Meilleurs paramètres : ", best_parameters)

""" 
Apèrs optimisation :
Précision :  ? %
"""
