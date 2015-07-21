
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[1]:

from Metadatos import generaPathProyecto, RMSLE
from CargaDatos import retornaSets
from Preparacion import preparaDf, separacionEntrenaObjetivo
from Modelo import definirModelo


# In[42]:

from CargaDatos import generaPath


# ### Definicion de Funciones

# # Ejecucion de rutina

# In[2]:

path_proyecto = generaPathProyecto()
sets_df = retornaSets(path_proyecto = path_proyecto)


# In[3]:

prepDf = preparaDf()


# In[4]:

df = prepDf.preparar(sets_df)


# In[6]:

df_envio = prepDf.preparar(sets_df, train_o_envio= 'test')


# In[30]:

df_envio.head(4)


# In[10]:

X_train, X_test, y_train, y_test = separacionEntrenaObjetivo(df, semilla = 1962, prop_prueba= 0.3)


# In[11]:

modelo = definirModelo()


# In[12]:

modelo.fit(X = X_train.values, y = y_train.values)


# In[13]:

modelo.best_score_


# In[14]:

RMSLE(y_train, modelo.predict(X_train))


# In[15]:

RMSLE(y_test, modelo.predict(X_test))


# In[17]:

X_envio = prepDf.preparar(sets_df, train_o_envio= 'test')


# In[18]:

y_envio = modelo.predict(X_envio.values)


# In[31]:

import pandas as pd


# In[37]:

y_envio = pd.DataFrame(data= y_envio, index= range(1,y_envio.shape[0] + 1), columns= ['cost'])
y_envio.index.name = 'id'


# In[41]:

path_proyecto


# In[44]:

generaPath(path_proyecto, 'y_envio')


# In[45]:

y_envio.to_csv(path_or_buf= generaPath(path_proyecto, 'y_envio.csv))


# In[ ]:

resultados = validarModelo(modelo)


# In[ ]:

registrarModelo(modelo)


# In[ ]:

generarEnvio(modelo, input)

