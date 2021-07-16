import numpy as np 
import pandas as pd

import numpy as np 
import pandas as pd

import os
from joblib import dump, load
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier

# ----------------- REGRESSÃO -----------------
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
# ---------------------------------------------

# --------------- CLASSIFICAÇÃO -----------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# ---------------------------------------------

def reg_knn_grid_search(X_train, y_train, score_metrics, cv, best_criteria):
    start_time = time.time()
    
    knn_hiperparameters = {
        'n_neighbors': range(5, 51, 5), 
        'weights': ['uniform', 'distance'], 
        'metric': ["euclidean", "manhattan", "chebyshev"]}

    knn = KNeighborsRegressor()
    knn_gs = GridSearchCV(knn, knn_hiperparameters, scoring=score_metrics, cv=cv, return_train_score=True, refit=best_criteria)
    knn_gs.fit(X_train, y_train)
    print('[DONE] Treino e geração da GridSearch do modelo KNN para Regressão --> {} segundos'.format(time.time() - start_time))
    
    return knn_gs
  
def reg_dt_grid_search(X_train, y_train, score_metrics, cv, best_criteria):
    start_time = time.time()
        
    dt_hiperparameters = {
        'criterion': ['mse', 'mae'],
        'splitter': ['best', 'random']
    }
    
    dt = DecisionTreeRegressor()
    dt_gs = GridSearchCV(dt, dt_hiperparameters, scoring=score_metrics, cv=cv, return_train_score=True, refit=best_criteria)
    dt_gs.fit(X_train, y_train)
    print('[DONE] Treino e geração da GridSearch do modelo Árvore de Decisão para Regressão --> {} segundos'.format(time.time() - start_time))
    
    return dt_gs
  
def svr_grid_search(X_train, y_train, score_metrics, cv, best_criteria):
    start_time = time.time()
    
    svr_hiperparameters = {
        'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
        'svr__gamma': ['scale', 'auto'], 
        'svr__C': [0.1, 1, 10, 100, 1000, 3000]}
        
    # scale data as recommended by: https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
    svr_pipeline = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
    svr_gs = GridSearchCV(svr_pipeline, svr_hiperparameters, scoring=score_metrics, cv=cv, return_train_score=True, refit=best_criteria)
    svr_gs.fit(X_train, y_train)
    print('[DONE] Treino e geração da GridSearch do modelo SVM para Regressão --> {} segundos'.format(time.time() - start_time))

    return svr_gs
  
def linreg_grid_search(X_train, y_train, score_metrics, cv, best_criteria):
    start_time = time.time()
    
    linreg_estimator = LinearRegression(normalize=True)
    linreg_gs = GridSearchCV(linreg_estimator, {}, scoring=score_metrics, cv=cv, return_train_score=True, refit=best_criteria)
    linreg_gs.fit(X_train, y_train)
    print('[DONE] Treino e geração da GridSearch do modelo Regressão Linear --> {} segundos'.format(time.time() - start_time))
    
    return linreg_gs

def ensure_directory_exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def save_models(base_dir, models):
    ensure_directory_exists(base_dir)
    
    for model_key in models:
        model_instance = models[model_key]
        dump(model_instance, base_dir + f"/{model_key}")

def load_models(base_dir):
    result = {}
    
    for model_key in os.listdir(base_dir):
        model_instance = load(base_dir + f"/{model_key}")
        result[model_key] = model_instance
    return result

def classif_dt_grid_search(X_train, y_train, score_metrics, cv, best_criteria):
    start_time = time.time()
    
    dt_hiperparameters = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random']
    }
    
    dt = DecisionTreeClassifier()
    dt_gs = GridSearchCV(dt, dt_hiperparameters, scoring=score_metrics, cv=cv, return_train_score=True, refit=best_criteria)
    dt_gs.fit(X_train, y_train)
    print('[DONE] Treino e geração da GridSearch do modelo Árvore de Decisão para Classificação --> {} segundos'.format(time.time() - start_time))
    
    return dt_gs

def naive_bayes_grid_search(X_train, y_train, score_metrics, cv, best_criteria):
    start_time = time.time()
    
    naive_bayes_hiperparameters = {'var_smoothing': [1e-7, 1e-8, 1e-9, 1e-10, 1e-11]}
    
    naive_bayes = GaussianNB()
    naive_bayes_gs = GridSearchCV(naive_bayes, naive_bayes_hiperparameters, scoring=score_metrics, cv=cv, return_train_score=True, refit=best_criteria)
    naive_bayes_gs.fit(X_train, y_train)
    print('[DONE] Treino e geração da GridSearch do modelo Naive Bayes para Classificação --> {} segundos'.format(time.time() - start_time))
    
    return naive_bayes_gs

def classif_knn_grid_search(X_train, y_train, score_metrics, cv, best_criteria):
    start_time = time.time()
    
    knn_hiperparameters = {
        'KNN__n_neighbors': range(5, 51, 5), 
        'KNN__weights': ['uniform', 'distance'], 
        'KNN__metric': ["euclidean", "manhattan", "chebyshev"]}
    
    knn = Pipeline([('scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])
    knn_gs = GridSearchCV(knn, knn_hiperparameters, scoring=score_metrics, cv=cv, return_train_score=True, refit=best_criteria)
    knn_gs.fit(X_train, y_train)
    print('[DONE] Treino e geração da GridSearch do modelo KNN para Classificação --> {} segundos'.format(time.time() - start_time))
    
    return knn_gs

def logreg_grid_search(X_train, y_train, score_metrics, cv, best_criteria):
    start_time = time.time()
    
    logreg_hiperparameters = {
        "LR__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sg', 'saga'], 
        "LR__penalty": ['none', 'l1', 'l2', 'elasticnet'], 
        "LR__C": [0.001, .009, 0.01, .09, 1, 5, 10, 25, 100, 250]}
    logreg = Pipeline([('scaler', StandardScaler()), ('LR', LogisticRegression())])
    logreg_gs = GridSearchCV(logreg, logreg_hiperparameters, scoring=score_metrics, cv=cv, return_train_score=True, refit=best_criteria)
    logreg_gs.fit(X_train, y_train)
    print('[DONE] Treino e geração da GridSearch do modelo Regressão Logística --> {} segundos'.format(time.time() - start_time))
    
    return logreg_gs

def svc_grid_search(X_train, y_train, score_metrics, cv, best_criteria):
    start_time = time.time()
    svc_hiperparameters = {
        'SVC__base_estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
        'SVC__base_estimator__gamma': ['scale', 'auto'], 
        'SVC__base_estimator__C': [0.1, 1, 10, 100]}
    # scale data as recommended by: https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
    
    n_estimators = 15
    svc_pipeline = Pipeline([('scaler', StandardScaler()), ('SVC', BaggingClassifier(SVC(), max_samples=1.0 / n_estimators, n_estimators=n_estimators))])
    
    svc_gs = GridSearchCV(svc_pipeline, svc_hiperparameters, scoring=score_metrics, cv=cv, return_train_score=True, refit=best_criteria)
    svc_gs.fit(X_train, y_train)
    print('[DONE] Treino e geração da GridSearch do modelo SVM para Classificação --> {} segundos'.format(time.time() - start_time))
    
    return svc_gs
    
def get_grid_results(gs, sort_metric, include_rank=False, include_params=True, ascending=False):
    grid_results = pd.DataFrame(gs.cv_results_).sort_values(by=f"rank_test_{sort_metric}", ascending=ascending)
    
    mean_columns = [col for col in grid_results.columns if col.startswith("mean_test")]
    rank_columns = [col for col in grid_results.columns if (col.startswith("rank_") and include_rank)]
    columns = mean_columns + rank_columns
    columns = columns + ["params"] if include_params else columns
    
    return grid_results[columns]

def treat_regression_columns(df, score_metrics_names):
    # para regressao, as metricas de erro vem com valor negativo, pois o sklearn 
    # multiplica por -1 o resultado das metricas, para que numeros maiores (mais proximos de 0)
    # sejam priorizados
    # a ideia, é desfazer essa multiplicação
    mean_columns = [f"mean_test_{name}" for name in score_metrics_names]
    df[mean_columns] *= -1


def train_regression_models(needs_to_train, models_path, X_train=None, y_train=None, score_metrics=None, cv=None, best_criteria=None, grids_path=None):
    if needs_to_train:
        reg_models = {
            "linear_regression": linreg_grid_search(X_train, y_train, score_metrics, cv, best_criteria),
            "knn": reg_knn_grid_search(X_train, y_train, score_metrics, cv, best_criteria),
            "decision_tree": reg_dt_grid_search(X_train, y_train, score_metrics, cv, best_criteria),
            "svr": svr_grid_search(X_train, y_train, score_metrics, cv, best_criteria)
        }
        best_regression_models = {k: v.best_estimator_ for k, v in reg_models.items()}
        
        print('\nSalvando os resultados das grids search de cada modelo...')
        for k, v in reg_models.items():
            results = get_grid_results(reg_models[k], best_criteria, ascending=True)
            negativated_metrics = list(score_metrics.keys())
            # r2 error ja vai ser positiva, visto que maior é melhor, naturalmente.
            negativated_metrics.remove("r2_error")
            treat_regression_columns(results, negativated_metrics)
            results.to_csv(grids_path + '/' + k + '.csv', index=False)
            
        print('Salvando os melhores modelos...')
        save_models(models_path, best_regression_models)
    else:
        print('Carregando os modelos...')
        best_regression_models = load_models(models_path)
        
    return best_regression_models

def train_classification_models(needs_to_train, models_path, X_train=None, y_train=None, score_metrics=None, cv=None, best_criteria=None, grids_path=None, label=None):
    if needs_to_train:
        classif_models = {
            (label + "_logistic_regression"): logreg_grid_search(X_train, y_train, score_metrics, cv, best_criteria),
            (label + "_knn"): classif_knn_grid_search(X_train, y_train, score_metrics, cv, best_criteria),
            (label + "_decision_tree"): classif_dt_grid_search(X_train, y_train, score_metrics, cv, best_criteria),
            (label + "_naive_bayes"): naive_bayes_grid_search(X_train, y_train, score_metrics, cv, best_criteria),
            (label + "_svc"): svc_grid_search(X_train, y_train, score_metrics, cv, best_criteria)
        }
        best_classification_models = {k: v.best_estimator_ for k, v in classif_models.items()}
        
        print('\nSalvando os resultados das grids search de cada modelo...')
        for k, v in classif_models.items():
            results = get_grid_results(classif_models[k], best_criteria, ascending=True)
            results.to_csv(grids_path + '/' + k + '.csv', index=False)
            
        print('Salvando os melhores modelos...')
        save_models(models_path, best_classification_models)
    else:
        print('Carregando os modelos...')
        best_classification_models = load_models(models_path)
        
    return best_classification_models