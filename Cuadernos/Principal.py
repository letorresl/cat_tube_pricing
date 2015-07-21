
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[1]:

from Metadatos import generaPathProyecto, RMSLE
from CargaDatos import retornaSets
from Preparacion import preparaDf, separacionEntrenaObjetivo
from Modelo import definirModelo


# ### Definicion de Funciones

# # Ejecucion de rutina

# In[2]:

path_proyecto = generaPathProyecto()
sets_df = retornaSets(path_proyecto = path_proyecto)
df = preparaDf(sets_df)


# In[3]:

X_train, X_test, y_train, y_test = separacionEntrenaObjetivo(df, semilla = 1962, prop_prueba= 0.3)


# In[4]:

modelo = definirModelo()


# In[5]:

modelo.fit(X = X_train.values, y = y_train.values)


# In[6]:

modelo.best_score_


# In[7]:

RMSLE(y_train, modelo.predict(X_train))


# In[8]:

RMSLE(y_test, modelo.predict(X_test))


# In[9]:

df_envio = preparaDf(sets_df, train_o_envio= 'test')


# In[13]:

set(df.columns).difference(set(df_envio.columns))


# In[14]:

set(df_envio.columns).difference(set(df.columns))


# In[11]:

df_envio.shape


# In[ ]:

resultados = validarModelo(modelo)


# In[ ]:

registrarModelo(modelo)


# In[ ]:

generarEnvio(modelo, input)

