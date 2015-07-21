
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[1]:

from Metadatos import generaPathProyecto, RMSLE
from CargaDatos import retornaSets
from Preparacion import preparaDf, separacionEntrenaObjetivo
from Modelo import definirModelo


# In[2]:

from CargaDatos import generaPath


# ### Definicion de Funciones

# # Ejecucion de rutina

# In[3]:

path_proyecto = generaPathProyecto()
sets_df = retornaSets(path_proyecto = path_proyecto)


# In[4]:

prepDf = preparaDf()


# In[5]:

df = prepDf.preparar(sets_df)


# In[6]:

X_train, X_test, y_train, y_test = separacionEntrenaObjetivo(df, semilla = 1962, prop_prueba= 0.3)


# In[7]:

modelo = definirModelo()


# In[17]:

y_train.values.reshape(y_train.shape[0],)


# In[18]:

modelo.fit(X = X_train.values, y = y_train.values.reshape(y_train.shape[0],))


# In[9]:

modelo.best_params_


# In[10]:

modelo.best_score_


# In[11]:

RMSLE(y_train, modelo.predict(X_train))


# In[12]:

RMSLE(y_test, modelo.predict(X_test))


# In[19]:

X_envio = prepDf.preparar(sets_df, train_o_envio= 'test')


# In[20]:

y_envio = modelo.predict(X_envio.values)


# In[21]:

import pandas as pd


# In[22]:

y_envio = pd.DataFrame(data= y_envio, index= range(1,y_envio.shape[0] + 1), columns= ['cost'])
y_envio.index.name = 'id'


# In[23]:

y_envio.to_csv(path_or_buf= generaPath(path_proyecto, 'y_envio3.csv'))


# In[ ]:

resultados = validarModelo(modelo)


# In[ ]:

registrarModelo(modelo)


# In[ ]:

generarEnvio(modelo, input)

