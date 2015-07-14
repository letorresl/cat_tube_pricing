
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[ ]:

import numpy as np
from pandas import read_excel, read_csv, merge
from os.path import join
from Metadatos import generaPathProyecto
from Metadatos import buscarMasReciente


# ### Definicion de funciones

# In[ ]:

def cargadatos(path_proyecto, nombre_archivo):
    # path de los metadatos
    path_metadatos = join(path_proyecto, 'Bases', 'Locales', nombre_archivo)
    df = read_csv(filepath_or_buffer = path_metadatos)
    return df


def cargaSets(path_proyecto, nombre_set, indice = None):
    path_set = join(path_proyecto, 'Bases', 'Locales')
    nombreX = buscarMasReciente(path_set, nombre_set)
    path_set = join(path_set, nombreX)
    df = read_csv(filepath_or_buffer = path_set, index_col = indice)
    return df


# In[1]:

def generaPath(path, nombre_archivo):
    path_unido = join(path, 'Bases', 'Locales', nombre_archivo)
    return path_unido


# In[ ]:

def retornaSets(path_proyecto):
    #train = cargadatos(path_proyecto, 'train_set.csv')
    train = read_csv(filepath_or_buffer = generaPath(path_proyecto, 'train_set.csv'), #index_col = 0,
                     true_values = ['Yes'], false_values = ['No'], parse_dates = ['quote_date'])
    #test = cargadatos(path_proyecto, 'test_set.csv')
    test = read_csv(filepath_or_buffer = generaPath(path_proyecto, 'test_set.csv'), index_col = 0,
                     true_values = ['Yes'], false_values = ['No'], parse_dates = ['quote_date'])
    #tube = cargadatos(path_proyecto, 'tube.csv')
    tube = read_csv(filepath_or_buffer = generaPath(path_proyecto, 'tube.csv'), #index_col = 0,
                    true_values = ['Y'], false_values = ['N'])
    #bill = cargadatos(path_proyecto, 'bill_of_materials.csv')
    bill = read_csv(filepath_or_buffer = generaPath(path_proyecto, 'bill_of_materials.csv'))#, index_col = 0)
    #specs = cargadatos(path_proyecto, 'comp_threaded.csv')
    specs = read_csv(filepath_or_buffer = generaPath(path_proyecto, 'comp_threaded.csv'))#, index_col = 0)

    components = cargadatos(path_proyecto, 'components.csv')
    comp_adaptor = cargadatos(path_proyecto, 'comp_adaptor.csv')
    comp_boss = cargadatos(path_proyecto, 'comp_boss.csv')
    comp_elbow = cargadatos(path_proyecto, 'comp_elbow.csv')
    comp_float = cargadatos(path_proyecto, 'comp_float.csv')
    comp_hfl = cargadatos(path_proyecto, 'comp_hfl.csv')
    comp_nut = cargadatos(path_proyecto, 'comp_nut.csv')
    components = cargadatos(path_proyecto, 'components.csv')
    comp_other = cargadatos(path_proyecto, 'comp_other.csv')
    comp_sleeve = cargadatos(path_proyecto, 'comp_sleeve.csv')
    comp_straight = cargadatos(path_proyecto, 'comp_straight.csv')
    comp_tee = cargadatos(path_proyecto, 'comp_tee.csv')
    comp_threaded = cargadatos(path_proyecto, 'comp_threaded.csv')
    tube_end_form = cargadatos(path_proyecto, 'tube_end_form.csv')
    type_component = cargadatos(path_proyecto, 'type_component.csv')
    type_connection = cargadatos(path_proyecto, 'type_connection.csv')
    type_end_form = cargadatos(path_proyecto, 'type_end_form.csv')
    
    diccionario = {'train' : train, 'test' : test, 'tube' : tube, 'bill' : bill, 'specs' : specs,
                  'comp_adaptor' : comp_adaptor, 'comp_boss' : comp_boss, 'comp_elbow' : comp_elbow,
                  'comp_float' : comp_float, 'comp_hfl' : comp_hfl, 'comp_other' : comp_other,
                  'comp_sleeve' : comp_sleeve, 'comp_straight' : comp_straight, 'comp_straight' : comp_straight,
                  'comp_tee' : comp_tee, 'comp_threaded' : comp_threaded, 'tube_end_form' : tube_end_form,
                  'type_component' : type_component, 'type_connection' : type_connection,
                  'type_end_form' : type_end_form, 'components' : components}
    
    return diccionario


# In[1]:

get_ipython().system(u'ls -lha ../Bases/Locales/')


# # Ejecucion de rutina
