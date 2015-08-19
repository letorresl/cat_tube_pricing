
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
    decisor = GradientBoostingRegressor(random_state = 1962, n_estimators= 4000,
                                        learning_rate= 0.02, max_features= 0.2,
                                        min_samples_leaf= 7, max_depth= 8)
    puntuador = RMSLE_score
    modelo = Pipeline(steps = [('imputador', imputador), ('escalador', escalador),
                                ('decisor', decisor)])

    return modelo


# # Ejecucion de rutina
