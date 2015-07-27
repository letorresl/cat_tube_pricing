
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[1]:

from Metadatos import generaPathProyecto, RMSLE
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

# In[21]:

def validarModelo():
    dScores = {'best' : -modelo.best_score_,
               'train' : RMSLE(y_train, np.exp(modelo.predict(X_train)) - 1),
               'test' : RMSLE(y_test, np.exp(modelo.predict(X_test)) - 1)}
    print str('La mejor puntuacion de la crossvalidacion fue: {}\n' +
              'La puntuacion del entrenamiento fue: {}\n' +
              'La puntuacion de la prueba fue: {}').format(dScores['best'], dScores['train'], dScores['test'])


# In[11]:

def logTransf(df):
    return np.log(df + 1)


# In[19]:

def antilogTransf(df):
    return np.exp(df) - 1


# # Ejecucion de rutina

# ### Entrenamiento

# In[6]:

path_proyecto = generaPathProyecto()
sets_df = retornaSets(path_proyecto = path_proyecto)


# In[8]:

prepDf = preparaDf()
df = prepDf.preparar(sets_df)
X_train, X_test, y_train, y_test = separacionEntrenaObjetivo(df, semilla = 1962, prop_prueba= 0.3)


# In[14]:

y_train = logTransf(y_train)
y_test = logTransf(y_test)


# In[20]:

y_train = antilogTransf(y_train)
y_test = antilogTransf(y_test)


# In[15]:

modelo = definirModelo()
modelo.fit(X = X_train.values, y = y_train.values.reshape(y_train.shape[0],))


# ### Presentacion de resultados

# In[16]:

modelo.best_params_


# In[22]:

validarModelo()


# ### Eliminacion de valores atipicos

# In[9]:

import pandas as pd


# In[10]:

vectErrores = (y_train - pd.DataFrame(data= modelo.predict(X_train), index= X_train.index, columns= ['cost']))**2


# In[11]:

cuartiles = vectErrores.quantile(q= [0.25, 0.75])


# In[12]:

ric = cuartiles.iloc[1, 0] - cuartiles.iloc[0, 0]


# In[13]:

cuartiles.iloc[0, 0] - 1.5*ric, cuartiles.iloc[1, 0] + 1.5*ric


# In[14]:

indice = vectErrores[vectErrores.cost < cuartiles.iloc[1, 0] + 1.5*ric].index


# In[15]:

X_train = X_train.ix[indice,]


# In[16]:

y_train = y_train.ix[indice,]


# In[36]:

gg.ggplot(gg.aes(x= 'cost'), data= vectErrores) + gg.geom_boxplot()


# ### Experimentacion

# In[33]:

a = modelo.best_estimator_.steps[2][1]


# In[43]:

impDf = pd.DataFrame(data= zip(X_train.columns, a.feature_importances_), columns= ['nombre', 'importancia'])


# In[62]:

gg.ggplot(gg.aes(x= 'nombre', y= 'importancia'), data= impDf[impDf.importancia >= 0.01]) +gg.geom_bar(stat= 'identity') + gg.theme(axis_text_x= gg.element_text(angle= 45))


# ### Envio

# In[23]:

X_envio = prepDf.preparar(sets_df, train_o_envio= 'test')
generarEnvio(modelo, X_envio)


# In[25]:

get_ipython().magic(u'pinfo validarModelo')

