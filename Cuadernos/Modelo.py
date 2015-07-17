
# coding: utf-8

# # Configuracion

# ## Carga de librerias

# In[1]:

from numpy import log
from numpy.random import seed
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


# ## Definicion de funciones

# In[2]:

def RMSLE(y, y_pred):
    return mean_squared_error(log(y + 1), log(y_pred + 1))**0.5


# In[ ]:

def definirModelo(num_procesos = -1):
    seed(1989)
    imputador = Imputer()
    escalador = StandardScaler()
    #selectorVar = SelectKBest(k = 30)
    decisor = DecisionTreeRegressor(random_state = 1962)
    puntuador = RMSLE
    tuberia = Pipeline(steps = [('imputador', imputador), ('escalador', escalador),
                                ('decisor', decisor)])
    # Definicion de parametros a ajustar en gridsearch
    parametros_tuberia = {'decisor__min_samples_split' : [1, 2, 4, 8, 16],
                          'decisor__min_samples_leaf' : [1, 2, 4, 8 , 16]}
    # Instanciacion de rejilla
    modelo = GridSearchCV(estimator = tuberia,
                          param_grid = parametros_tuberia,
                          scoring = puntuador,
                          n_jobs = num_procesos)
    # Ajuste de rejilla
    return modelo


# # Ejecucion de rutina
