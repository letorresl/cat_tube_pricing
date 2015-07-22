
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


# In[35]:

import ggplot as gg


# In[48]:

get_ipython().magic(u'matplotlib inline')


# ### Definicion de funciones

# In[ ]:

def validarModelo():
    dScores = {'best' : modelo.best_score_,
               'train' : RMSLE(y_train, modelo.predict(X_train)),
               'test' : RMSLE(y_test, modelo.predict(X_test))}
    print str('La mejor puntuacion de la crossvalidacion fue: {}\n' +
              'La puntuacion del entrenamiento fue: {}\n' +
              'La puntuacion de la prueba fue: {}').format(dScores**)


# # Ejecucion de rutina

# ### Entrenamiento

# In[3]:

path_proyecto = generaPathProyecto()
sets_df = retornaSets(path_proyecto = path_proyecto)


# In[4]:

prepDf = preparaDf()
df = prepDf.preparar(sets_df)
X_train, X_test, y_train, y_test = separacionEntrenaObjetivo(df, semilla = 1962, prop_prueba= 0.3)


# In[7]:

modelo = definirModelo()
modelo.fit(X = X_train.values, y = y_train.values.reshape(y_train.shape[0],))


# ### Presentacion de resultados

# In[ ]:

validarModelo()


# In[33]:

a = modelo.best_estimator_.steps[2][1]


# In[43]:

impDf = pd.DataFrame(data= zip(X_train.columns, a.feature_importances_), columns= ['nombre', 'importancia'])


# In[62]:

gg.ggplot(gg.aes(x= 'nombre', y= 'importancia'), data= impDf[impDf.importancia >= 0.01]) +gg.geom_bar(stat= 'identity') + gg.theme(axis_text_x= gg.element_text(angle= 45))


# ### Envio

# In[19]:

X_envio = prepDf.preparar(sets_df, train_o_envio= 'test')
generarEnvio(modelo, X_envio)


# In[ ]:

resultados = validarModelo(modelo)

