import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Carregar o dataset
url = 'https://hastie.su.domains/ElemStatLearn/datasets/prostate.data'
data = pd.read_csv(url, delimiter='\t')

# Padronização dos atributos de entrada para que eles tenham média 0 e variância 1
features = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
X = data[features]
y = data['lpsa']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão do dataset em dois conjuntos, treinamento e teste, conforme indicado nos índices da última coluna
train_indices = data['train'] == 'T'
test_indices = data['train'] == 'F'
X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Definição dos parâmetros para GridSearchCV
param_grid = {
    'alpha': np.linspace(0, 5000, 100),
    'fit_intercept': [False]
}
kf = KFold(n_splits=10, shuffle=True, random_state=3)
ridge = Ridge()

# Realização do GridSearchCV com validação cruzada
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)

# Resultados do GridSearchCV
results = grid_search.cv_results_

# Obtenção dos scores médios e desvios padrão
mean_scores = -results['mean_test_score']
std_scores = results['std_test_score']
lambdas = param_grid['alpha']

# Regra de 1 desvio padrão
min_score = np.min(mean_scores)
min_score_index = np.argmin(mean_scores)
min_score_std = std_scores[min_score_index]
min_score_lambda = lambdas[min_score_index]
lambda_1se_index = np.where(mean_scores <= min_score + min_score_std)[0][-1]
lambda_1se = lambdas[lambda_1se_index]
lambda_1se_score = mean_scores[lambda_1se_index]

# Plotagem da curva de validação cruzada
plt.figure()
plt.plot(lambdas, mean_scores, label='MSE (Ridge)')
plt.scatter(min_score_lambda, min_score, c='r', label='Menor score', zorder=3)
plt.scatter(lambda_1se, lambda_1se_score, c='g', label='Melhor lambda', zorder=3)
plt.fill_between(lambdas, min_score, min_score + min_score_std, alpha=0.2, label='01 desvio padrão')
plt.xlabel('Lambda')
plt.ylabel('CV MSE')
plt.title('Curva de Validação Cruzada do Ridge')
plt.legend()
plt.show()

# Exibição do melhor lambda
best_lambda_ridge = grid_search.best_params_['alpha']
print(f'Melhor lambda (Ridge): {best_lambda_ridge}')

# Treinamento do modelo final com o melhor lambda
final_ridge_model = Ridge(alpha=best_lambda_ridge)
final_ridge_model.fit(X_train, y_train)
y_pred_test_final_ridge = final_ridge_model.predict(X_test)
mse_test_final_ridge = mean_squared_error(y_test, y_pred_test_final_ridge)
print(f'MSE - Teste (Final Ridge): {mse_test_final_ridge}')
