
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

# In[ ]:

path_proyecto = generaPathProyecto()
sets_df = retornaSets(path_proyecto = path_proyecto)
df = preparaDf(sets_df)


# In[94]:

X_train, X_test, y_train, y_test = separacionEntrenaObjetivo(df, semilla = 1962, prop_prueba= 0.3)


# In[95]:

modelo = definirModelo()


# In[96]:

modelo.fit(X = X_train.values, y = y_train.values)


# In[97]:

modelo.best_score_


# In[98]:

RMSLE(y_train, modelo.predict(X_train))


# In[99]:

RMSLE(y_test, modelo.predict(X_test))


# In[ ]:

resultados = validarModelo(modelo)


# In[ ]:

registrarModelo(modelo)


# In[ ]:

generarEnvio(modelo, input)

