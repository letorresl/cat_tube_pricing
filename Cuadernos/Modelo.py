
# coding: utf-8

# # Configuracion

# ## Carga de librerias

# In[1]:

from numpy.random import seed
from sklearn.metrics import make_scorer
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
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
    decisor = GradientBoostingRegressor(random_state = 1962, n_estimators= 8000)
    puntuador = RMSLE_score
    tuberia = Pipeline(steps = [('imputador', imputador), ('escalador', escalador),
                                ('decisor', decisor)])
    # Definicion de parametros a ajustar en gridsearch
    parametros_tuberia = {'decisor__min_samples_leaf' : [8 , 16],
                          'decisor__max_depth' : [8, 12],
                          'decisor__learning_rate' : [0.005, 0.002],
                          'decisor__max_features' : [0.4, 0.3, 0.2],
                          'decisor__subsample' : [1.0, 0.7, 0.6]}
    # Instanciacion de rejilla
    modelo = GridSearchCV(estimator = tuberia,
                          param_grid = parametros_tuberia,
                          scoring = puntuador,
                          n_jobs = num_procesos)
    # Ajuste de rejilla
    return modelo


# # Ejecucion de rutina

# In[4]:

get_ipython().magic(u'pinfo GradientBoostingRegressor')

