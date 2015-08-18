
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


# In[3]:

from numpy import power, log1p, expm1
from sklearn.base import BaseEstimator
import xgboost as xgb


# ## Definicion de funciones

# In[4]:

class WichoregressorMixin(object):

    def score(self, X, y, sample_weight=None):
        """
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        RMSLE
        """
        
        RMSLE_score = make_scorer(RMSLE, greater_is_better=False)

        return RMSLE_score(y, self.predict(X), sample_weight=sample_weight)


# In[5]:

class XgboostRegressor(BaseEstimator, WichoregressorMixin):
    """Predicts the majority class of its training data."""
    
    def __init__(self,
                 objective= 'reg:linear',
                 eta= 0.02,
                 min_child_weight= 6,
                 subsample= 0.7,
                 colsample_bytree= 0.6,
                 scale_pos_weight= 0.8,
                 silent= 1,
                 max_depth= 8,
                 max_delta_step= 2,
                 nthread= 1):
        
        self.objective = objective
        self.eta = eta
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.silent = silent
        self.max_depth = max_depth
        self.max_delta_step = max_delta_step
        self.nthread = nthread
        self.modelos = []

        
    def fit(self, X, y):
        params = {}
        modelos = []
        
        params["objective"] = self.objective
        params["eta"] = self.eta
        params["min_child_weight"] = self.min_child_weight
        params["subsample"] = self.subsample
        params["colsample_bytree"] = self.colsample_bytree
        params["scale_pos_weight"] = self.scale_pos_weight
        params["silent"] = self.silent
        params["max_depth"] = self.max_depth
        params["max_delta_step"]= self.max_delta_step
        plst = list(params.items())
        
        labels = log1p(y)
        xgX = xgb.DMatrix(X, label=labels)
        for num_rounds in [2000, 3000, 4000]:
            modelos.append(xgb.train(plst, xgX, num_rounds))
        
        labels = power(y, 1.0/16.0)
        xgX = xgb.DMatrix(X, label=labels)
        num_rounds = 4000        
        modelos.append(xgb.train(plst, xgX, num_rounds))
        
        self.modelos = modelos
        
        return self
    
    
    def predict(self, X):
        xgX = xgb.DMatrix(X)
        
        preds1 = self.modelos[0].predict(xgX)
        preds2 = self.modelos[1].predict(xgX)
        preds3 = self.modelos[2].predict(xgX)
        preds4 = self.modelos[3].predict(xgX)
        preds = (0.58*expm1( (preds1+preds2+preds3)/3))+(0.42*power(preds4,16))
        
        return preds


# In[6]:

def definirModelo(num_procesos = -1):
    seed(1989)
    RMSLE_score = make_scorer(RMSLE, greater_is_better=False)
    
    imputador = Imputer()
    escalador = StandardScaler()
    #selectorVar = SelectKBest(k = 30)
    decisor = XgboostRegressor()
    puntuador = RMSLE_score
    tuberia = Pipeline(steps = [('imputador', imputador), ('escalador', escalador),
                                ('decisor', decisor)])
    # Definicion de parametros a ajustar en gridsearch
    parametros_tuberia = {'decisor__colsample_bytree' : [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]}
    # Instanciacion de rejilla
    modelo = GridSearchCV(estimator = tuberia,
                          param_grid = parametros_tuberia,
                          scoring = puntuador,
                          n_jobs = num_procesos)
    # Ajuste de rejilla
    return modelo


# # Ejecucion de rutina
