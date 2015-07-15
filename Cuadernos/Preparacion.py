
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[1]:

from pandas import merge


# ### Definicion de funciones

# In[24]:

def integracion1(sets_df, train_o_envio = 'train'):
    col_llave = 'tube_assembly_id'
    df = merge(left = sets_df[train_o_envio], right = sets_df['bill'], how = 'left', on = col_llave)
    df = merge(left = df, right = sets_df['tube'], how = 'left', on = col_llave)
    return df


# # Ejecucion de rutina

# def integracion2(sets_df, df):
#     col_llave = 

# from Metadatos import generaPathProyecto
# from CargaDatos import retornaSets
# path_proyecto = generaPathProyecto()
# sets_df = retornaSets(path_proyecto)
# df = integracion1(sets_df = sets_df)

# df.component_id_1.unique().shape

# df.columns

# sets_df['components']['component_type_id'].unique().shape[0]

# sets_df['components'].columns
