
# coding: utf-8

# # Configuracion

# ## Carga de librerias

# In[1]:

from pandas import DataFrame


# In[ ]:

from datetime import datetime
from os.path import join


# In[2]:

from Metadatos import generaPathProyecto, antilogTransf


# ## Definicion de funciones

# In[ ]:

def generarEnvio(modelo, X_envio):
    y_envio = modelo.predict(X_envio.values)
    y_envio.cost = antilogTransf(y_envio.cost)
    y_envio = DataFrame(data= y_envio, index= range(1,y_envio.shape[0] + 1), columns= ['cost'])
    y_envio.index.name = 'id'
    guardarDf(y_envio, 'y_envio', True)


# In[ ]:

def guardarDf(dataFrame, nombreArchivo = None, guardarIndice = False):
    stringHora = datetime.today().strftime('%Y%m%d_%H%M%S')
    if nombreArchivo is None:
        nombreArchivo = 'df_'
    pathGuardado = join(generaPathProyecto(), 'Bases',
                        'Locales', str(nombreArchivo + '_' + stringHora + '.txt'))
    dataFrame.to_csv(path_or_buf = pathGuardado, index = guardarIndice)


# # Ejecucion de rutina
