
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


# In[70]:

columnas = impDf[impDf.importancia >= 0.01].nombre


# In[72]:

X_train = X_train[columnas]


# In[74]:

X_envio = X_envio[columnas]


# In[6]:

X_train, X_test, y_train, y_test = separacionEntrenaObjetivo(df, semilla = 1962, prop_prueba= 0.3)


# In[7]:

modelo = definirModelo()


# In[17]:

y_train.values.reshape(y_train.shape[0],)


# In[75]:

modelo.fit(X = X_train.values, y = y_train.values.reshape(y_train.shape[0],))


# In[76]:

modelo.best_params_


# In[77]:

modelo.best_score_


# In[78]:

RMSLE(y_train, modelo.predict(X_train))


# In[80]:

X_test = X_test[columnas]


# In[81]:

RMSLE(y_test, modelo.predict(X_test))


# In[33]:

a = modelo.best_estimator_.steps[2][1]


# In[35]:

import ggplot as gg


# In[43]:

impDf = pd.DataFrame(data= zip(X_train.columns, a.feature_importances_), columns= ['nombre', 'importancia'])


# In[48]:

get_ipython().magic(u'matplotlib inline')


# In[62]:

gg.ggplot(gg.aes(x= 'nombre', y= 'importancia'), data= impDf[impDf.importancia >= 0.01]) +gg.geom_bar(stat= 'identity') + gg.theme(axis_text_x= gg.element_text(angle= 45))


# In[19]:

X_envio = prepDf.preparar(sets_df, train_o_envio= 'test')


# In[82]:

X_envio = X_envio[columnas]


# In[83]:

y_envio = modelo.predict(X_envio.values)


# In[84]:

import pandas as pd


# In[85]:

y_envio = pd.DataFrame(data= y_envio, index= range(1,y_envio.shape[0] + 1), columns= ['cost'])
y_envio.index.name = 'id'


# In[86]:

y_envio.to_csv(path_or_buf= generaPath(path_proyecto, 'y_envio4.csv'))


# In[ ]:

resultados = validarModelo(modelo)


# In[ ]:

registrarModelo(modelo)


# In[ ]:

generarEnvio(modelo, input)

