
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[23]:

import numpy as np
from pandas import read_excel, read_csv, merge
from os.path import join
from Metadatos import generaPathProyecto
from Metadatos import buscarMasReciente


# ### Definicion de funciones

# In[2]:

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


# In[3]:

def generaPath(path_proyecto, nombre_archivo):
    return join(path_proyecto, 'Bases', 'Locales', nombre_archivo)


# In[24]:

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
    
    return train, test, tube, bill, specs, comp_adaptor, comp_boss, comp_elbow,
    comp_float, comp_hfl, comp_nut, comp_other, comp_sleeve, comp_straight,
    comp_tee, comp_threaded, tube_end_form, type_component, type_connection,
    type_end_form


# # Ejecucion de rutina

# train, test, tube, bill, specs, comp_adaptor, comp_boss, comp_elbow,
# comp_float, comp_hfl, comp_nut, comp_other, comp_sleeve, comp_straight,
# comp_tee, comp_threaded, tube_end_form, type_component, type_connection,
# type_end_form = retornaSets(path_proyecto)

# from Metadatos import generaPathProyecto

# path_proyecto = generaPathProyecto()

# #train = cargadatos(path_proyecto, 'train_set.csv')
# train = read_csv(filepath_or_buffer = generaPath(path_proyecto, 'train_set.csv'), #index_col = 0,
#                  true_values = ['Yes'], false_values = ['No'], parse_dates = ['quote_date'])
# #test = cargadatos(path_proyecto, 'test_set.csv')
# test = read_csv(filepath_or_buffer = generaPath(path_proyecto, 'test_set.csv'), index_col = 0,
#                  true_values = ['Yes'], false_values = ['No'], parse_dates = ['quote_date'])
# #tube = cargadatos(path_proyecto, 'tube.csv')
# tube = read_csv(filepath_or_buffer = generaPath(path_proyecto, 'tube.csv'), #index_col = 0,
#                 true_values = ['Y'], false_values = ['N'])
# #bill = cargadatos(path_proyecto, 'bill_of_materials.csv')
# bill = read_csv(filepath_or_buffer = generaPath(path_proyecto, 'bill_of_materials.csv'))#, index_col = 0)
# #specs = cargadatos(path_proyecto, 'comp_threaded.csv')
# specs = read_csv(filepath_or_buffer = generaPath(path_proyecto, 'comp_threaded.csv'))#, index_col = 0)
# 
# comp_adaptor = cargadatos(path_proyecto, 'comp_adaptor.csv')
# comp_boss = cargadatos(path_proyecto, 'comp_boss.csv')
# comp_elbow = cargadatos(path_proyecto, 'comp_elbow.csv')
# comp_float = cargadatos(path_proyecto, 'comp_float.csv')
# comp_hfl = cargadatos(path_proyecto, 'comp_hfl.csv')
# comp_nut = cargadatos(path_proyecto, 'comp_nut.csv')
# components = cargadatos(path_proyecto, 'components.csv')
# comp_other = cargadatos(path_proyecto, 'comp_other.csv')
# comp_sleeve = cargadatos(path_proyecto, 'comp_sleeve.csv')
# comp_straight = cargadatos(path_proyecto, 'comp_straight.csv')
# comp_tee = cargadatos(path_proyecto, 'comp_tee.csv')
# comp_threaded = cargadatos(path_proyecto, 'comp_threaded.csv')
# tube_end_form = cargadatos(path_proyecto, 'tube_end_form.csv')
# type_component = cargadatos(path_proyecto, 'type_component.csv')
# type_connection = cargadatos(path_proyecto, 'type_connection.csv')
# type_end_form = cargadatos(path_proyecto, 'type_end_form.csv')

# train.columns

# bill.columns

# tube.columns

# import pandas as pd

# def fase1(df):
#     col_llave = 'tube_assembly_id'
#     df_unido = pd.merge(left = df, right = bill, right_on = col_llave, left_on = col_llave)
#     return df_unido
# 
# def fase2(df):
#     col_llave = 'tube_assembly_id'
#     df_unido = pd.merge(left = df, right = tube, right_on = col_llave, left_on = col_llave)
#     return df_unido

# fase2(fase1(train))

# type_end_form.head(4)

# total = set()
# for i in range(1, 9):
#     total = total.union(set(bill.ix[:, 'component_id_%d' % i]))

# comp_adaptor.columns

# comp_boss.columns

# comp_float.columns

# type_connection.shape

# len(comp_adaptor.component_id.unique()) +\
# len(comp_boss.component_id.unique()) +\
# len(comp_elbow.component_id.unique()) +\
# len(comp_float.component_id.unique()) +\
# len(comp_hfl.component_id.unique()) +\
# len(comp_nut.component_id.unique()) +\
# len(comp_other.component_id.unique()) +\
# len(comp_sleeve.component_id.unique()) +\
# len(comp_straight.component_id.unique()) +\
# len(comp_tee.component_id.unique()) +\
# len(comp_threaded.component_id.unique())

# import pandas as pd

# pd.merge(left = train, right = bill, how = 'left', left_on = 'tube_assembly_id', right_on = 'tube_assembly_id').columns

# if __name__ == '__main__':
#     tube = pd.read_csv('../input/tube.csv',
#                        index_col=0,
#                        true_values=['Y'],
#                        false_values=['N'])
# 
#     materials = pd.read_csv('../input/bill_of_materials.csv', index_col=0)
#     specs = pd.read_csv('../input/specs.csv', index_col=0)
# 
#     for idx in tube.index:
#         if tube.ix[idx, 'material_id'] is not np.nan:
#             tube.ix[idx, tube.ix[idx, 'material_id']] = 1
#         for i in range(1, 11):
#             if i < 9 and materials.ix[idx, 'component_id_%d' % i] is not np.nan:
#                 tube.ix[idx, materials.ix[idx, 'component_id_%d' % i]] = \
#                     materials.ix[idx, 'quantity_%d' % i]
#             if specs.ix[idx, 'spec%d' % i] is not np.nan:
#                 tube.ix[idx, specs.ix[idx, 'spec%d' % i]] = 1
#     tube.drop('material_id', inplace=True, axis=1)
#     tube.fillna(0, inplace=True)
#     tube.to_csv('tube_extended.csv')
