import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# Carregar o dataset
# data = pd.read_csv('prostatedata.txt', delimiter='\t')
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

# (c) Encontre o modelo linear de regressão ótimo no critério de mínimos quadrados (solução LS)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_train_linear = linear_model.predict(X_train)
y_pred_test_linear = linear_model.predict(X_test)
mse_train_linear = mean_squared_error(y_train, y_pred_train_linear)
mse_test_linear = mean_squared_error(y_test, y_pred_test_linear)
print(f'MSE - Treinamento (LS): {mse_train_linear}')
print(f'MSE - Teste (LS): {mse_test_linear}')

# (d) Implementação modelos lineares regularizados pelos métodos Ridge e Lasso com lambda = 0.25
lambda_val = 0.25
ridge_model = Ridge(alpha=lambda_val)
ridge_model.fit(X_train, y_train)
y_pred_train_ridge = ridge_model.predict(X_train)
y_pred_test_ridge = ridge_model.predict(X_test)
lasso_model = Lasso(alpha=lambda_val)
lasso_model.fit(X_train, y_train)
y_pred_train_lasso = lasso_model.predict(X_train)
y_pred_test_lasso = lasso_model.predict(X_test)
mse_train_ridge = mean_squared_error(y_train, y_pred_train_ridge)
mse_test_ridge = mean_squared_error(y_test, y_pred_test_ridge)
mse_train_lasso = mean_squared_error(y_train, y_pred_train_lasso)
mse_test_lasso = mean_squared_error(y_test, y_pred_test_lasso)
print(f'MSE - Treinamento (Ridge): {mse_train_ridge}')
print(f'MSE - Teste (Ridge): {mse_test_ridge}')
print(f'MSE - Treinamento (Lasso): {mse_train_lasso}')
print(f'MSE - Teste (Lasso): {mse_test_lasso}')

# (e) Aplicação do k-fold cross-validation para selecionar o melhor valor de lambda
def plot_cv_curve(model_class, X, y, lambdas, model_name):
    # k-fold cross-validation
    mean_scores = list()
    std_scores = list()
    kf = KFold(n_splits=10, shuffle=True, random_state=3)
    for alpha in lambdas:
        scores = cross_val_score(model_class(alpha=alpha), X, y, cv=kf, scoring='neg_mean_squared_error')
        mean_scores.append(-scores.mean())
        std_scores.append(scores.std())
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    # Regra de 1 desvio padrão
    min_score = np.min(mean_scores)
    min_score_index = np.argmin(mean_scores)
    min_score_std = std_scores[min_score_index]/np.sqrt(10)
    min_score_lambda = lambdas[min_score_index]
    lambda_1se_index = np.where(mean_scores <= min_score + min_score_std)[0][-1]
    lambda_1se = lambdas[lambda_1se_index]
    lambda_1se_score = mean_scores[lambda_1se_index]
    # plot
    plt.figure()
    plt.plot(lambdas, mean_scores, label=f'MSE ({model_name})')
    plt.scatter(lambda_1se, lambda_1se_score, c='g', label='Melhor lambda', zorder=3)
    plt.fill_between(lambdas, min_score - min_score_std, min_score + min_score_std, alpha=0.2, label='Faixa para buscar o lambda')
    plt.errorbar(min_score_lambda, min_score, yerr=min_score_std, fmt='o', color='r', label='Menor score e seu desvio padrão')

    plt.xlabel('Lambda')
    # plt.xscale('log')
    plt.ylabel('CV MSE')
    plt.title(f'Curva de Validação Cruzada do {model_name}')
    plt.legend()
    plt.show()
    return lambda_1se

best_lambda_ridge = plot_cv_curve(Ridge, X_train, y_train, np.linspace(0.0001,100,100), 'Ridge')
best_lambda_lasso = plot_cv_curve(Lasso, X_train, y_train, np.linspace(0.0001,0.5,100), 'Lasso')

print(f'Melhor lambda (Ridge): {best_lambda_ridge}')
print(f'Melhor lambda (Lasso): {best_lambda_lasso}')

# (f) Calculando o MSE para o conjunto de teste
final_ridge_model = Ridge(alpha=best_lambda_ridge)
final_ridge_model.fit(X_train, y_train)
final_lasso_model = Lasso(alpha=best_lambda_lasso)
final_lasso_model.fit(X_train, y_train)
y_pred_test_final_ridge = final_ridge_model.predict(X_test)
y_pred_test_final_lasso = final_lasso_model.predict(X_test)
mse_test_final_ridge = mean_squared_error(y_test, y_pred_test_final_ridge)
mse_test_final_lasso = mean_squared_error(y_test, y_pred_test_final_lasso)
print(f'MSE - Teste (Final Ridge): {mse_test_final_ridge}')
print(f'MSE - Teste (Final Lasso): {mse_test_final_lasso}')

# (g) (Bônus) Estimando o desvio padrão dos coeficientes do modelo pelo método de bootstrap dos resíduos
def bootstrap_residuals(model, X_train, y_train, n_bootstrap=1000):
    residuals = y_train - model.predict(X_train)
    coefs = []
    for _ in range(n_bootstrap):
        sampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
        y_bootstrap = model.predict(X_train) + sampled_residuals
        model.fit(X_train, y_bootstrap)
        coefs.append(model.coef_)
    coefs = np.array(coefs)
    return np.std(coefs, axis=0)

bootstrap_std_linear = bootstrap_residuals(linear_model, X_train, y_train)
bootstrap_std_ridge = bootstrap_residuals(final_ridge_model, X_train, y_train)
bootstrap_std_lasso = bootstrap_residuals(final_lasso_model, X_train, y_train)
print(f'Desvio padrão dos coeficientes (LS): {bootstrap_std_linear}')
print(f'Desvio padrão dos coeficientes (Ridge): {bootstrap_std_ridge}')
print(f'Desvio padrão dos coeficientes (Lasso): {bootstrap_std_lasso}')
