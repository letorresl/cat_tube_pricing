
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[1]:

from pandas import merge, DataFrame, DatetimeIndex
from numpy import floor
from numpy.random import choice, seed
from sklearn.preprocessing import label_binarize


# In[2]:

from IPython.core.debugger import Tracer
tracer = Tracer()


# ### Definicion de funciones

# In[3]:

def limpieza2(df):
    # limpieza de NaN's en las columnas de component_id_n
    #return df.dropna(axis = 0, how = 'all', subset = ['component_id_{}'.format(i) for i in range(1, 9)])
    cols_componentes = ['component_id_{}'.format(i) for i in range(1, 9)]
    df[cols_componentes] = df[cols_componentes].fillna('')
    return df


# In[59]:

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
    
    def vectSumaMultCol(self, df, lista_cols, nombre_base, eliminarColOriginal = True, clases= None):
        # Definir un df vacio para albergar el resultado de la vectorizacion
        resultado = None
        # Establecer las clases que tendra la vectorizacion
        if clases is None:
            clases = set()
            for columna in lista_cols:
                clases = clases.union(set(df[columna].dropna(axis= 0)))
        # Iterar para cada columna del conjunto que integran la vectorizacion suma
        for columna in lista_cols:
            # Seleccionar los datos con los cuales se conformara la vectorizacion
            col_a_vectorizar = df[columna].dropna(axis= 0)
            # Vectorizar la columna y agregarlo a la matriz de resultados
            if col_a_vectorizar.shape[0] != 0:
                vectorizacion = DataFrame(data= label_binarize(col_a_vectorizar, classes= list(clases)),
                                          index= col_a_vectorizar.index,
                                          columns= ['{}__{}'.format(nombre_base, clase) for clase in clases])      
                if resultado is not None:
                    resultado = resultado.add(vectorizacion, axis= 0, fill_value= 0)
                else:
                    resultado = vectorizacion
        if eliminarColOriginal:
                df = df.drop(labels= lista_cols, axis= 1)
        #return merge(left= df, right= resultado, how= 'left', left_index= True,
        #             right_index= True, copy= False)
        return resultado
    
    def multipleUnionVectorizada(self, df_i, df_d, lista_conexion, l_col_llaves_i, llave_d,
                                 prefijo_vectorizacion):
        # Obtener las distintas clases en que se separara la matriz
        clases = set()
        for columna in lista_conexion:
            clases = clases.union(set(df_d[columna].dropna(axis= 0)))
        resultado = None
        # Unir el dataframe df con 'specs' con la columna 'component_id' para 
        # cada col de 'df' 'component_id_#'
        for col in l_col_llaves_i:
            df_conn = merge(left= df_i, right= df_d[[llave_d] + lista_conexion], how= 'left',
                            left_on= col, right_on= llave_d)
            df_conn.index = df_i.index
            # A partir de las 'connection_type_id_#' obtenidas de tal union, obtener la matriz vectorizacion de la variable
            # connection.
            m_vect = self.vectSumaMultCol(df_conn, lista_conexion, prefijo_vectorizacion, clases= clases)
            # Almacenar la matriz vectorizacion tras agregarla a la matriz anterior
            if resultado is not None:
                if m_vect is not None:
                    resultado = resultado.add(m_vect, axis= 0, fill_value= 0)
            else:
                resultado = m_vect
        return resultado

    def preparar(self, sets_df,  train_o_envio = 'train'):
        df = self.integracion1(sets_df = sets_df,  train_o_envio =  train_o_envio)
        df = merge(left= df, right= self.integracion3(sets_df= sets_df, df= df), how= 'left', left_index= True,
                   right_index= True, copy= False)
        df['year'] = DatetimeIndex(df.quote_date).year
        df['month'] = DatetimeIndex(df.quote_date).month
        df['day'] = DatetimeIndex(df.quote_date).day
        ### experimentacion
        prefijo_vectorizacion = 'connection_type_id'
        lista_conexion = ['connection_type_id_{}'.format(i) for i in range(1, 5)]
        l_col_llaves_i = ['component_id_{}'.format(i) for i in range(1, 9)]
        llave_d = 'component_id'
        v_connect = self.multipleUnionVectorizada(df_i= df, df_d= sets_df['specs'], lista_conexion= lista_conexion,
                                                  l_col_llaves_i= l_col_llaves_i, llave_d = llave_d,
                                                  prefijo_vectorizacion= prefijo_vectorizacion)
        df = merge(left= df, right= v_connect, how= 'left', left_index= True,
                   right_index= True, copy= False)
        ## fin experimentacion
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
            df = self.vectorizacion(df, columna, eliminarColOriginal = True,
                                    clases= self.matrices_vectorizadas[columna] if train_o_envio == 'test' else None)
        df = self.reordenaCols(df)
        df = df.astype('float')
        return df


# In[5]:

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

# prepDf = preparaDf()
# df = prepDf.preparar(sets_df)
