
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[1]:

from Metadatos import generaPathProyecto, RMSLE, logTransf, antilogTransf
from CargaDatos import retornaSets
from Preparacion import preparaDf, separacionEntrenaObjetivo
from Modelo import definirModelo
from CargaDatos import generaPath
from Salida import generarEnvio, guardarModelo, generarY


# In[2]:

import ggplot as gg


# In[3]:

get_ipython().magic(u'matplotlib inline')


# In[4]:

import numpy as np


# ### Definicion de funciones

# In[5]:

def validarModelo():
    dScores = {'best' : -modelo.best_score_,
               'train' : RMSLE(y_train, antilogTransf(modelo.predict(X_train))),
               'test' : RMSLE(y_test, antilogTransf(modelo.predict(X_test)))}
    print str('La mejor puntuacion de la crossvalidacion fue: {}\n' +
              'La puntuacion del entrenamiento fue: {}\n' +
              'La puntuacion de la prueba fue: {}').format(dScores['best'], dScores['train'], dScores['test'])


# # Ejecucion de rutina

# ### Entrenamiento

# In[6]:

path_proyecto = generaPathProyecto()
sets_df = retornaSets(path_proyecto = path_proyecto)


# In[7]:

prepDf = preparaDf()
df = prepDf.preparar(sets_df)
X_train, X_test, y_train, y_test = separacionEntrenaObjetivo(df, semilla = 1962, prop_prueba= 0.3)


# In[ ]:

modelo = definirModelo()


# In[ ]:

import matplotlib.pyplot as plt
#from sklearn.datasets import load_digits
#from sklearn.svm import SVC
from sklearn.learning_curve import validation_curve
from sklearn.metrics import make_scorer

#digits = load_digits()
X, y = X_train.values, logTransf(y_train).values.reshape(y_train.shape[0],)
RMSLE_score = make_scorer(RMSLE, greater_is_better=False)


param_range = [0.02, 0.01, 0.005]
train_scores, test_scores = validation_curve(
    modelo, X, y, param_name="decisor__learning_rate", param_range=param_range,
    cv=10, scoring=RMSLE_score, n_jobs=4)
train_scores, test_scores = -train_scores, -test_scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with GBR")
plt.xlabel("$\n_estimators$")
plt.ylabel("Score")
plt.ylim(0.0, 0.25)
plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="g")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
plt.legend(loc="best")
plt.show()


# In[8]:

modelo.fit(X = X_train.values, y = logTransf(y_train).values.reshape(y_train.shape[0],))


# ### Presentacion de resultados

# In[11]:

modelo.best_params_


# In[12]:

validarModelo()


# ### Almacenamiento modelo y y_estimadas

# In[2]:

guardarModelo(modelo, path_proyecto)
generarY(modelo, X_train, X_test)


# ### Experimentacion

# ### Envio

# In[13]:

X_envio = prepDf.preparar(sets_df, train_o_envio= 'test')
generarEnvio(modelo, X_envio)

