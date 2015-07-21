
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[1]:

from pandas import merge, DataFrame
from numpy import floor
from numpy.random import choice, seed
from sklearn.preprocessing import label_binarize


# In[19]:

#from IPython.core.debugger import Tracer
#tracer = Tracer()


# ### Definicion de funciones

# In[ ]:

def limpieza2(df):
    # limpieza de NaN's en las columnas de component_id_n
    #return df.dropna(axis = 0, how = 'all', subset = ['component_id_{}'.format(i) for i in range(1, 9)])
    cols_componentes = ['component_id_{}'.format(i) for i in range(1, 9)]
    df[cols_componentes] = df[cols_componentes].fillna('')
    return df


# In[ ]:

class preparaDf:
    def __init__(self):
        self.matrices_vectorizadas = {}
    
    def integracion1(self, sets_df, train_o_envio = 'train'):
        col_llave = 'tube_assembly_id'
        df = merge(left = sets_df[train_o_envio], right = sets_df['bill'], how = 'left', on = col_llave)
        df = merge(left = df, right = sets_df['tube'], how = 'left', on = col_llave)
        return df


    def integracion3(self, sets_df, df):
        #tracer()
        resultado = None
        # Establecer las clases que tendra la vectorizacion
        tipos_comp = sorted(sets_df['components'].component_type_id.unique())
        # establecer los datos de los componentes
        df_tipo_comp = sets_df['components'][['component_id', 'component_type_id']]
        # Iterar para cada columna componente
        for i in range(1, 9):
            #if i == 2:
                #tracer()
            # la cantidad y componente correspondientes para la iteracion
            cantidad, componente = 'quantity_{}'.format(str(i)), 'component_id_{}'.format(str(i))
            # Agregar al df de entrenamiento la columna de "tipo de componente" para el componente_i
            entrenamiento = df[[componente]].reset_index().dropna(axis= 0, how= 'all', subset= [componente])
            no_vect = merge(left = entrenamiento, right = df_tipo_comp, how = 'left', left_on = componente,
                            right_on = 'component_id', copy = False).set_index('index')
            # Vectorizar el tipo de componente
            # AQUI ESTA EL PROBLEMA -->
            if no_vect.shape[0] != 0:
                vectorizacion = DataFrame(data= label_binarize(no_vect.component_type_id, classes=tipos_comp),
                                          index= no_vect.index,
                                          columns= ['{}__{}'.format('components', comp_id) for comp_id in tipos_comp])
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

    def vectorizacion(self, df, nombCol, eliminarColOriginal = True, clases= None):
        # limpieza de renglones sin valores
        df_temp = df[[nombCol]].reset_index().dropna(axis= 0, how= 'all', subset= [nombCol]).set_index('index')
        # establecimiento de las clases de la variable nominal
        if clases is None:
            clases = sorted(df_temp[nombCol].unique())
            self.matrices_vectorizadas[nombCol] = clases
        # creacion de la binarizacion estandar
        array_binarizado = label_binarize(df_temp[nombCol], classes= clases)
        # dataframe de la binarizacion respetando el indice
        array_binarizado = DataFrame(data= array_binarizado, index= df_temp.index,
                                     columns= ['{}__{}'.format(nombCol, str(clase)) for clase in clases])
        if eliminarColOriginal:
            df = df.drop(labels= nombCol, axis= 1)
        return merge(left= df, right= array_binarizado, how= 'left', left_index= True,
                     right_index= True, copy= False)

    def reordenaCols(self, df):
        columnas = df.columns
        return df[sorted(columnas)]

    def preparar(self, sets_df,  train_o_envio = 'train'):
        df = self.integracion1(sets_df = sets_df,  train_o_envio =  train_o_envio)
        df = merge(left= df, right= self.integracion3(sets_df= sets_df, df= df), how= 'left', left_index= True,
                   right_index= True, copy= False)
        df = df.drop(labels= [u'component_id_1', u'quantity_1', u'component_id_2', u'quantity_2', 
                              u'component_id_3', u'quantity_3', u'component_id_4', u'quantity_4',
                              u'component_id_5', u'quantity_5', u'component_id_6', u'quantity_6',
                              u'component_id_7', u'quantity_7', u'component_id_8', u'quantity_8',
                              u'tube_assembly_id',
                              # IMPORTANTE: eliminar la fecha de la cotizacion es una medida temporal
                              # PORHACER: transformar la variable fecha de cotizacion a una tipo
                              # float (por ejemplo, una variable para year, otra para month, etc)
                              u'quote_date'], axis= 1)
        for columna in ['supplier', 'material_id', 'end_a', 'end_x', 'other']:
            df = self.vectorizacion(df, columna,
                               self.matrices_vectorizadas[columna] if train_o_envio == 'test' else None)
        df = self.reordenaCols(df)
        df = df.astype('float')
        return df


# In[9]:

def separacionEntrenaObjetivo(df, semilla, prop_prueba = 0.30):
    seed(semilla)
    # SEPARACION SETS ENTRENAMIENTO/PRUEBA
    # permutacion de indices
    if prop_prueba != 0:
        tam_prueba = int(floor(df.shape[0] * prop_prueba))
        indices_prueba = choice(a= df.index.values, size= tam_prueba, replace= False)
        # Valores
        X_train = df[~df.index.isin(indices_prueba)].drop(labels = ['cost'],
                                                          axis = 1)
        X_test = df[df.index.isin(indices_prueba)].drop(labels = ['cost'],
                                                        axis = 1)
        # resultados
        y_train = df[~df.index.isin(indices_prueba)][['cost']]
        y_test = df[df.index.isin(indices_prueba)][['cost']]
    else:
        # Valores
        X_train = df.drop(labels = ['cost'], axis = 1)
        X_test = None
        # resultados
        y_train = df[['cost']]
        y_test = None
    return X_train, X_test, y_train, y_test


# # Ejecucion de rutina

# from Metadatos import generaPathProyecto
# from CargaDatos import retornaSets
# path_proyecto = generaPathProyecto()
# sets_df = retornaSets(path_proyecto)

# df = integracion1(sets_df = sets_df)
# df = limpieza2(df)
# df = merge(left= df, right= integracion3(sets_df= sets_df, df= df), how= 'left', left_index= True,
#            right_index= True, copy= False)
# df.drop(labels= [u'component_id_1', u'quantity_1', u'component_id_2', u'quantity_2', 
#                  u'component_id_3', u'quantity_3', u'component_id_4', u'quantity_4',
#                  u'component_id_5', u'quantity_5', u'component_id_6', u'quantity_6',
#                  u'component_id_7', u'quantity_7', u'component_id_8', u'quantity_8',
#                  u'tube_assembly_id'],
#         axis= 1, inplace= True)
# df = vectorizacion(df, 'supplier')
# df = vectorizacion(df, 'material_id')
# df = vectorizacion(df, 'end_a')
# df = vectorizacion(df, 'end_x')
# df = vectorizacion(df, 'other')
