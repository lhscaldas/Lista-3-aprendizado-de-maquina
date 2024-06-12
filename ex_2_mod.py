import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Carregar o dataset
url = 'https://hastie.su.domains/ElemStatLearn/datasets/prostate.data'
data = pd.read_csv(url, delimiter='\t')

# (a) Padronização dos atributos de entrada para que eles tenham média 0 e variância 1
features = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
X = data[features]
y = data['lpsa']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (b) Divisão o dataset em dois conjuntos, treinamento e teste, conforme indicado nos índices da última coluna
train_indices = data['train'] == 'T'
test_indices = data['train'] == 'F'
X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

def plot_cv_curve(model, X, y, lambdas, model_name):
    # k-fold cross-validation
    mean_scores = list()
    std_scores = list()
    for alpha in lambdas:
        model.alpha = alpha
        kf = KFold(n_splits=10, shuffle=True, random_state=43)
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        mean_scores.append(-scores.mean())
        std_scores.append(scores.std())
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    # Regra de 1 desvio padrão
    min_score = np.min(mean_scores)
    min_score_index = np.argmin(mean_scores)
    min_score_std = std_scores[min_score_index]
    min_score_lambda = lambdas[min_score_index]
    lambda_1se_index = np.where(mean_scores <= min_score + min_score_std)[0][-1]
    lambda_1se = lambdas[lambda_1se_index]
    lambda_1se_score = mean_scores[lambda_1se_index]
    # plot
    plt.figure()
    plt.plot(lambdas, mean_scores, label=f'MSE ({model_name})')
    plt.scatter(min_score_lambda, min_score, c='r', label='Menor score', zorder=3)
    plt.scatter(lambda_1se, lambda_1se_score, c='g', label='Melhor lambda', zorder=3)
    plt.fill_between(lambdas, min_score, min_score + min_score_std, alpha=0.2, label='01 desvio padrão')
    plt.xlabel('Lambda')
    # plt.xscale('log')
    plt.ylabel('CV MSE')
    plt.title(f'Curva de Validação Cruzada do {model_name}')
    plt.legend()
    plt.show()
    return lambda_1se

best_lambda_ridge = plot_cv_curve(Ridge(), X_train, y_train, np.linspace(0,20,50), 'Ridge')
print(f'Melhor lambda (Ridge): {best_lambda_ridge}')
