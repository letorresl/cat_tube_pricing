
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[1]:

from pandas import merge


# In[52]:

from sklearn.preprocessing import label_binarize
from pandas import DataFrame


# ### Definicion de funciones

# In[27]:

def integracion1(sets_df, train_o_envio = 'train'):
    col_llave = 'tube_assembly_id'
    df = merge(left = sets_df[train_o_envio], right = sets_df['bill'], how = 'left', on = col_llave)
    df = merge(left = df, right = sets_df['tube'], how = 'left', on = col_llave)
    return df


# In[80]:

def limpieza2(df):
    # limpieza de NaN's en las columnas de component_id_n
    return df.dropna(axis = 0, how = 'all', subset = ['component_id_{}'.format(i) for i in range(1, 9)])


# In[232]:

from IPython.core.debugger import Tracer
trace = Tracer()


# In[456]:

def integracion3(sets_df, df):
    resultado = None
    # Establecer las clases que tendra la vectorizacion
    tipos_comp = sorted(sets_df['components'].component_type_id.unique())
    # establecer los datos de los componentes
    df_tipo_comp = sets_df['components'][['component_id', 'component_type_id']]
    # Iterar para cada columna componente
    for i in range(1, 9):
        # la cantidad y componente correspondientes para la iteracion
        cantidad, componente = 'quantity_{}'.format(str(i)), 'component_id_{}'.format(str(i))
        # Agregar al df de entrenamiento la columna de "tipo de componente" para el componente_i
        entrenamiento = df[[componente]].reset_index().dropna(axis= 0, how= 'all', subset= [componente])
        no_vect = merge(left = entrenamiento, right = df_tipo_comp, how = 'left', left_on = componente,
                        right_on = 'component_id', copy = False).set_index('index')
        # Vectorizar el tipo de componente
        vectorizacion = DataFrame(data= label_binarize(no_vect.component_type_id, classes=tipos),
                                 index= no_vect.index, columns= tipos)
        # Multiplicar la matriz de la vectorizacion por la cantidad solicitada
        vectorizacion = vectorizacion.mul(df.loc[no_vect.index, cantidad], axis = 0)
        # Sumar el producto a la la suma de las matrices de vectorizacion
        #try:
        #    resultado += vectorizacion
        #except UnboundLocalError:
        #    resultado = vectorizacion        
        if resultado is not None:
            resultado = resultado.add(vectorizacion, axis= 0, fill_value= 0)
        else:
            resultado = vectorizacion
        #trace()
    return resultado


# In[496]:

'{}__{}'.format('algo', '1')


# In[505]:

def vectorizacion(df, nombCol, eliminarColOriginal = True):
    # limpieza de renglones sin valores
    df_temp = df[[nombCol]].reset_index().dropna(axis= 0, how= 'all', subset= [nombCol]).set_index('index')
    # establecimiento de las clases de la variable nominal
    clases = sorted(df_temp[nombCol].unique())
    # creacion de la binarizacion estandar
    array_binarizado = label_binarize(df_temp[nombCol], classes= clases)
    # dataframe de la binarizacion respetando el indice
    array_binarizado = DataFrame(data= array_binarizado, index= df_temp.index,
                                 columns= ['{}__{}'.format(nombCol, str(clase)) for clase in clases])
    df.drop(labels= nombCol, axis= 0, inplace= True)
    return merge(left= df, right= array_binarizado, how= 'left', left_index= True,
                 right_index= True, copy= False)


# # Ejecucion de rutina

# In[501]:

from Metadatos import generaPathProyecto
from CargaDatos import retornaSets
path_proyecto = generaPathProyecto()
# sets_df = retornaSets(path_proyecto)
df = integracion1(sets_df = sets_df)
df = limpieza2(df)


# In[502]:

df = merge(left= df, right= integracion3(sets_df= sets_df, df= df), how= 'left', left_index= True,
           right_index= True, copy= False)


# In[503]:

df.drop(labels= [u'component_id_1', u'quantity_1', u'component_id_2', u'quantity_2', 
                 u'component_id_3', u'quantity_3', u'component_id_4', u'quantity_4',
                 u'component_id_5', u'quantity_5', u'component_id_6', u'quantity_6',
                 u'component_id_7', u'quantity_7', u'component_id_8', u'quantity_8'],
        axis= 1, inplace= True)


# In[ ]:

vectorizacion(df, 'other')

