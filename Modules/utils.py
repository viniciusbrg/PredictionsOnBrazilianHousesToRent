import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Biblioteca que permite plots mais robustos que o plt
import matplotlib.ticker as ticker 
from sklearn.metrics import mean_squared_error

def countplot_grid(data, cols, title):
    if(len(cols) < 4):
        fig, axs = plt.subplots(1, 3, figsize=(22, 10))
        fig.suptitle(title + '\nCount = Contagem de aluguéis')
        i = [0, 1, 2]
    else:
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle(title + '\nCount = Contagem de aluguéis')
        i = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for index, col in enumerate(cols):
        plt.setp(axs[i[index]].get_xticklabels(), rotation=80)
        sns.countplot(ax=axs[i[index]], data=data, x=col, palette='RdBu_r')
    plt.show()

def plot_corr_matrix(data, cols, title):
    plt.figure(figsize=(16, 8))
    correlation = data[cols].corr()
    mask = np.zeros_like(correlation, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(correlation, cmap='RdBu_r', annot=True, annot_kws={'size': 12}, mask=mask)
    plt.title(title, fontsize=18)
    plt.show()
  
def plot_distribution_large_data(data, title, xaxis_interval):
    '''
    Imprime um boxplot próprio para datasets grandes
    '''
    sns.set(style='whitegrid', font_scale=1.3)
    plt.figure(figsize=(16, 8))
    plt.title(title, fontsize=18)
    ax = sns.boxenplot(data=data, palette='PuBu', saturation=1, scale='area', orient='h')
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xaxis_interval))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.setp(ax.get_xticklabels(), rotation=45);

def boxenplot_continuos_features(data, cols):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    
    for index, ax in enumerate(axs.flat):
        col = cols[1:-1][index]
        sns.boxenplot(data=data[col], ax=ax, palette='PuBu', saturation=1, orient='h')
        ax.set_title('Distribuição dos valores da coluna {}'.format(col), fontsize=18)
        plt.setp(ax.get_xticklabels(), rotation=45);
    plt.tight_layout()
    plt.show()
  
def boxenplot_grid_quantitative_features(data, cols, suptitle):
    i = []
    if(len(cols) < 4):
        fig, axs = plt.subplots(1, 3, figsize=(20, 10),
                      subplot_kw={'xticks': [], 'yticks': []})
    else:
        fig, axs = plt.subplots(2, 2, figsize=(20, 20),
                      subplot_kw={'xticks': [], 'yticks': []})
        
    fig.suptitle(suptitle, fontsize=20, y=1.05)
    for index, ax in enumerate(axs.flat):
        sns.boxplot(ax=ax, x=cols[index], y='total (R$)', data=data, palette='RdBu_r')
        ax.set_title('Para a coluna {}'.format(cols[index]), fontsize=18)
        plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()
    
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)
    
def check_determinism(n):
    """ Teste pra ver se o k-fold do Scikit-learn é deterministico """
    
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    base = list(cv.split(X_train, y_train))
    for _ in range(n):
        l = list(cv.split(X_train, y_train))
        for i in range(len(l)):
            if (base[i][0] != l[i][0]).any():
                print("Not equal")
                return
            
def bar_values(ax, labels):
    rects = ax.patches
    
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 0.01, label, ha='center', va='bottom', fontsize=14, rotation=60)
    
    return ax

def plot_histogram_labels(x_label, y_label, labels, hue_label=None, data=None, palette='RdGy_r', title=None, ylabel_title=None, width=None, height=None):
    sns.set(style='whitegrid', font_scale=1.3)
    plt.figure(figsize=(width, height))
    
    ax = sns.barplot(x_label, y_label, hue=hue_label, data=data, palette=palette, ci=None)
    
    plt.title(title, fontsize=18)
    plt.xlabel('')
    plt.ylabel(ylabel_title, fontsize=14)
    plt.legend(loc='lower right')
    
    bar_values(ax, labels)
    plt.show()

def plot_labels_distribution(title, X, X_train, X_test, labels):
    data = {
      'positive': X[X == 1].shape[0],
      'negative': X[X == 0].shape[0]
    }
    
    train_data = {
      'positive': X_train[X_train == 1].shape[0],
      'negative': X_train[X_train == 0].shape[0]
    }
    
    test_data = {
      'positive': X_test[X_test == 1].shape[0],
      'negative': X_test[X_test == 0].shape[0]
    }
    
    classif_train_test = list(data.values())
    classif_train_test.extend(list(train_data.values()))
    classif_train_test.extend(list(test_data.values()))
    
    inform_train_test = pd.DataFrame(columns=['class', 'freq', 'set'])

    for label, value in zip((labels*3), classif_train_test):
        inform_train_test = inform_train_test.append({
            'class': label,
            'freq': value
        }, ignore_index=True)

    inform_train_test['set'][0:2] = 'data'
    inform_train_test['set'][2:4] = 'train'
    inform_train_test['set'][4:6] = 'test'
    
    plot_histogram_labels("class", "freq", classif_train_test, 'set', inform_train_test, 'RdYlBu', title, 'Count', width=12, height=12)
    
def save_sets(X, y, path):
    df_data = pd.DataFrame(columns=['X', 'y'])
    
    for x, y in zip(X, y):
        df_data = df_data.append({
            'X': x,
            'y': y
        }, ignore_index=True)
    
    df_data.to_json(path, orient='records')
    
def plot_errors_densities_by_model(models, X_test, y_test):
    errors_by_model = {
        name: np.abs(model.predict(X_test) - y_test)
        for name, model in models.items()
    }
    
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Gráficos - Densidade de Erros", fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.1)
    for i, model_name in enumerate(errors_by_model.keys()):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.set_title(model_name.replace('_', ' '))
        ax.set_yticks([])
        sns.distplot(errors_by_model[model_name], hist=False, kde=True,
                     kde_kws={'linewidth': 3},
                     label='airline',
                     color='darksalmon')
                     
def compute_scores_best_models(X_test, y_test, best_models, metrics, metric_names):
    model_scores = {}
    
    for model in best_models:
        model_instance = best_models[model]
        y_pred = model_instance.predict(X_test)
    
        scores = {k: v(y_test, y_pred) for k, v in zip(metric_names, metrics)}
        model_scores[model] = scores
    
    return model_scores

def classification_score_data(classif_scores):
    metric_names = {
    'acc': 'Accuracy',
    'p': 'Precision',
    'r': 'Recall',
    'f1': 'F1-score'
    }
    sorter_index = dict(zip(list(metric_names.values()), range(len(list(metric_names.values())))))
    
    score_data = pd.DataFrame(columns=['Score', 'Metric', 'Model'])
    for model, scores in classif_scores.items():
        for k, v in metric_names.items():
            score_data = score_data.append({
                'Model': ' '.join(model.split("_")[1:]),
                'Score': float('{:.2f}'.format(scores[k])),
                'Metric': v
            }, ignore_index=True)
    
    score_data['Metric_Rank'] = score_data['Metric'].map(sorter_index)
    score_data = score_data.sort_values(by=['Metric_Rank', 'Model'])
    
    return score_data