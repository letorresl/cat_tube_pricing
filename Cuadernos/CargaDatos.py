
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[ ]:

import numpy as np
from pandas import read_excel, read_csv
from os.path import join
from MetaDatos import generaPathProyecto
from MetaDatos import buscarMasReciente


# ### Definicion de funciones

# In[ ]:

def cargaMetadatos(path_proyecto):
    # path de los metadatos
    path_metadatos = join(path_proyecto, 'Bases', 'Locales', 'metaData_taxistandsID_name_GPSlocation.csv')
    df = read_csv(filepath_or_buffer = path_metadatos)
    return df

def cargaTrain(path_proyecto):
    # path de entrenamiento
    path_train = join(path_proyecto, 'Bases', 'Locales', 'train.csv.gz')
    df = read_csv(filepath_or_buffer = path_train, compression = 'gzip')
    return df

def cargaEnvio(path_proyecto):
    # path de entrenamiento
    path_envio = join(path_proyecto, 'Bases', 'Locales', 'test.csv.gz')
    df = read_csv(filepath_or_buffer = path_envio, compression = 'gzip')
    return df

def cargaSets(path_proyecto, nombre_set, indice = None):
    path_set = join(path_proyecto, 'Bases', 'Locales')
    nombreX = buscarMasReciente(path_set, nombre_set)
    path_set = join(path_set, nombreX)
    df = read_csv(filepath_or_buffer = path_set, index_col = indice)
    return df


# # Ejecucion de rutina
