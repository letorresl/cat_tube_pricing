
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[24]:

from pandas import merge


# ### Definicion de funciones

# In[26]:

def integracion1(sets_df, train_o_envio = 'train'):
    col_llave = 'tube_assembly_id'
    df = merge(left = sets_df[train_o_envio], right = sets_df['bill'], left_on = col_llave, right_on = col_llave)
    df = merge(left = df, right = sets_df['tube'], left_on = col_llave, right_on = col_llave)
    return df


# # Ejecucion de rutina

# from Metadatos import generaPathProyecto
# from CargaDatos import retornaSets
# path_proyecto = generaPathProyecto()
# sets_df = retornaSets(path_proyecto)
# df = integracion1(sets_df = sets_df)
# sets_df.keys()
# df.component_id_1.unique()
