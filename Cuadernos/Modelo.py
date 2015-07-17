
# coding: utf-8

# # Configuracion

# ## Carga de librerias

# In[ ]:

from sklearn.metrics import make_scorer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


# ## Definicion de funciones

# In[ ]:

hsMed = make_scorer(hsMedio, greater_is_better=False)
# Definicion del proceso del modelo

imputador = Imputer()
escalador = StandardScaler()
#selectorVar = SelectKBest(k = 30)
#decisor = GradientBoostingClassifier()
#decisor = LogisticRegression()
decisor = RandomForestRegressor(random_state = 1962)
puntuador = hsMed
tuberia = Pipeline(steps = [('imputador', imputador), ('escalador', escalador),
                            ('decisor', decisor)])
# Definicion de parametros a ajustar en gridsearch
parametros_tuberia = {'decisor__n_estimators' : [15, 20, 30],
                     'decisor__min_samples_split' : [1, 2, 4, 8, 16],
                     'decisor__min_samples_leaf' : [1, 2, 4, 8 , 16]}
# Instanciacion de rejilla
modelo = GridSearchCV(estimator = tuberia,
                      param_grid = parametros_tuberia,
                      scoring = puntuador,
                      n_jobs = -1)
# Ajuste de rejilla
modelo.fit(X = X_train.values, y = y_train.values)


# # Ejecucion de rutina
