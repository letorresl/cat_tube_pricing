
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[7]:

from Metadatos import generaPathProyecto, RMSLE
from CargaDatos import retornaSets
from Preparacion import preparaDf, separacionEntrenaObjetivo
from Modelo import definirModelo


# In[8]:

from CargaDatos import generaPath


# ### Definicion de Funciones

# # Ejecucion de rutina

# In[9]:

path_proyecto = generaPathProyecto()
sets_df = retornaSets(path_proyecto = path_proyecto)


# In[10]:

prepDf = preparaDf()


# In[11]:

df = prepDf.preparar(sets_df)


# In[12]:

X_train, X_test, y_train, y_test = separacionEntrenaObjetivo(df, semilla = 1962, prop_prueba= 0.3)


# In[13]:

modelo = definirModelo()


# In[14]:

modelo.fit(X = X_train.values, y = y_train.values)


# In[24]:

modelo.best_params_


# In[15]:

modelo.best_score_


# In[16]:

RMSLE(y_train, modelo.predict(X_train))


# In[17]:

RMSLE(y_test, modelo.predict(X_test))


# In[18]:

X_envio = prepDf.preparar(sets_df, train_o_envio= 'test')


# In[19]:

y_envio = modelo.predict(X_envio.values)


# In[20]:

import pandas as pd


# In[21]:

y_envio = pd.DataFrame(data= y_envio, index= range(1,y_envio.shape[0] + 1), columns= ['cost'])
y_envio.index.name = 'id'


# In[23]:

y_envio.to_csv(path_or_buf= generaPath(path_proyecto, 'y_envio2.csv'))


# In[ ]:

resultados = validarModelo(modelo)


# In[ ]:

registrarModelo(modelo)


# In[ ]:

generarEnvio(modelo, input)

