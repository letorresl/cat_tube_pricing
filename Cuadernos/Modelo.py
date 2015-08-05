
# coding: utf-8

# # Configuracion

# ## Carga de librerias

# In[1]:

from numpy.random import seed
from sklearn.metrics import make_scorer
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


# In[2]:

from Metadatos import RMSLE


# ## Definicion de funciones

# In[3]:

def definirModelo(num_procesos = -1):
    seed(1989)
    RMSLE_score = make_scorer(RMSLE, greater_is_better=False)
    
    imputador = Imputer()
    escalador = StandardScaler()
    #selectorVar = SelectKBest(k = 30)
    decisor = RandomForestRegressor(random_state = 1962, oob_score= True,
                                    n_estimators= 200, min_samples_split= 1, min_samples_leaf= 1)
    puntuador = RMSLE_score
    tuberia = Pipeline(steps = [('imputador', imputador), ('escalador', escalador),
                                ('decisor', decisor)])
    # Definicion de parametros a ajustar en gridsearch
    parametros_tuberia = {'decisor__max_features' : [1.0, 0.95, 0.90, 0.85]}
    # Instanciacion de rejilla
    modelo = GridSearchCV(estimator = tuberia,
                          param_grid = parametros_tuberia,
                          scoring = puntuador,
                          n_jobs = num_procesos)
    # Ajuste de rejilla
    return modelo


# # Ejecucion de rutina
