import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Carregar o dataset
data = pd.read_csv('prostatedata.txt', delimiter='\t')

# (a) Padronização dos atributos de entrada para que eles tenham média 0 e variância 1
features = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
X = data[features]
y = data['lpsa']
scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# (b) Divisão o dataset em dois conjuntos, treinamento e teste, conforme indicado nos índices da última coluna
train_indices = data['train'] == 'T'
test_indices = data['train'] == 'F'
# X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
X_train, X_test = X[train_indices], X[test_indices]
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train, y_test = y[train_indices], y[test_indices]

# (e) Aplicação do k-fold cross-validation para selecionar o melhor valor de lambda
def plot_cv_curve(model, X, y, lambdas, model_name):
    lambdas = np.linspace(0,10,50)
    mean_scores = list()
    std_scores = list()
    for alpha in lambdas:
        model.alpha = alpha
        scores_one_alpha = list() 
        for i in range(1):
            kf = KFold(n_splits=10, shuffle=True, random_state=i)
            scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
            scores_one_alpha.append(-scores)
        scores_one_alpha = np.array(scores_one_alpha)
        mean_scores.append(scores_one_alpha.mean())
        std_scores.append(scores_one_alpha.std())   
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    plt.figure()
    plt.plot(lambdas, mean_scores, label=f'{model_name} Mean CV MSE')
    # plt.fill_between(lambdas, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)
    # plt.xlabel('Lambda')
    plt.ylabel('Mean CV MSE')
    plt.title(f'{model_name} Cross-Validation Curve')
    plt.legend()
    plt.show()
    # Regra de 1 desvio padrão
    min_score = np.min(mean_scores)
    lambda_1se = lambdas[np.where(mean_scores <= min_score + std_scores[np.argmin(mean_scores)])[0][-1]]
    return lambda_1se

best_lambda_ridge = plot_cv_curve(Ridge(), X_train, y_train, np.logspace(0, 3, 100), 'Ridge')
print(f'Best lambda (Ridge): {best_lambda_ridge}')
