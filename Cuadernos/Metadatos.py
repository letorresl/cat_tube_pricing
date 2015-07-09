
# coding: utf-8

# # Configuracion

# ### Carga de librerias

# In[ ]:

from os.path import join, realpath, curdir, isfile
from os import listdir
from datetime import datetime
from regex import compile as comp


# ### Definicion de funciones

# In[ ]:

def buscarMasReciente(path_busqueda, nombre):
    # definir lista de archivos coincidentes
    coincidentes = []
    # definir el patron de busqueda
    patronFecha = comp(str(nombre + '[0-9]{8}_[0-9]{6}\.txt'))
    patronInstante = comp('[0-9]{8}_[0-9]{6}')
    # iterar sobre los archivos en el path
    for nombre_archivo in listdir(path_busqueda):
        # descartar los nombres que sean directorios
        if not isfile(join(path_busqueda, nombre_archivo)):
            continue
        # comprobar si el archivo coincide con el patron de busqueda
        if patronFecha.match(nombre_archivo) is not None:
            # almacenar en una lista el instante de tiempo marcado en el nombre del archivo
            instante = patronInstante.search(nombre_archivo).group()
            coincidentes.append(datetime.strptime(instante, '%Y%m%d_%H%M%S'))
    # obtener el maximo instante de tiempo de tiempo de la lista
    instante_max = max(coincidentes)
    return str(nombre + instante_max.strftime('%Y%m%d_%H%M%S') + '.txt')


# In[ ]:

def generaPathProyecto():
    path_proyecto = join(realpath(curdir), '..')
    return path_proyecto.split('Cuadernos')[0]


# # Ejecucion de rutina
