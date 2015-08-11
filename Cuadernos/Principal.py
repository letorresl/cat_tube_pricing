
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[1]:

from Metadatos import generaPathProyecto, RMSLE, logTransf, antilogTransf
from CargaDatos import retornaSets
from Preparacion import preparaDf, separacionEntrenaObjetivo
from Modelo import definirModelo
from CargaDatos import generaPath
from Salida import generarEnvio


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


# In[ ]:

prepDf = preparaDf()
df = prepDf.preparar(sets_df)
X_train, X_test, y_train, y_test = separacionEntrenaObjetivo(df, semilla = 1962, prop_prueba= 0.3)


# In[ ]:

modelo = definirModelo()
modelo.fit(X = X_train.values, y = logTransf(y_train).values.reshape(y_train.shape[0],))


# ### Presentacion de resultados

# In[13]:

modelo.best_params_


# In[ ]:

modelo.best_score


# In[12]:

validarModelo()


# ### Experimentacion

# ### Envio

# In[14]:

X_envio = prepDf.preparar(sets_df, train_o_envio= 'test')
generarEnvio(modelo, X_envio)

