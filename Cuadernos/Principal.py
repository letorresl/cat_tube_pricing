
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[ ]:

from Metadatos import generaPathProyecto


# In[ ]:

from CargaDatos import cargaSets


# In[3]:

from Preparacion import preparaDf


# In[2]:

from Modelo import definirModelo


# ### Definicion de Funciones

# # Ejecucion de rutina

# In[ ]:

path_proyecto = generaPathProyecto()


# In[ ]:

df = cargaTrain(path_proyecto = path_proyecto)


# df = mejorarCalidad(df)

# In[ ]:

X_train, X_test, y_train, y_test = preparacionDf(df, semilla = 1989)


# In[ ]:

modelo = definirModelo()


# In[ ]:

modelo.fit(X = X_train.values, y = y_train.values)


# In[ ]:

resultados = validarModelo(modelo)


# In[ ]:

registrarModelo(modelo)


# In[ ]:

generarEnvio(modelo, input)

