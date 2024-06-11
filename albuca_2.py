import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Carregar o DataFrame com os dados filtrados
dfProstata = pd.read_csv('prostatedata.txt', delimiter='\t')


# Separar os atributos (entradas) e o alvo (target)
X = dfProstata[['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']]
y = dfProstata['lpsa']

# Dividir os dados em treinamento (70%) e validação (30%)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.73, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_validation_std = scaler.transform(X_validation)

# Defina os valores de lambda que você deseja testar
lambda_values = np.linspace(0, 5, 50)

# Listas para armazenar os resultados do erro de validação
validation_scores_ridge = []

# Loop através dos valores de lambda
for lambda_val in lambda_values:
    # Treine e valide o modelo Ridge com lambda_val usando validação cruzada k-fold
    ridge_model = Ridge(alpha=lambda_val)
    ridge_scores = -cross_val_score(ridge_model, X_train_std, y_train, cv= 10, scoring='neg_mean_squared_error')
    validation_scores_ridge.append(ridge_scores.mean())

# Encontre o valor de lambda que resulta no MSE mínimo
best_lambda_ridge = lambda_values[np.argmin(validation_scores_ridge)]

# Treine o modelo Ridge com o melhor lambda
final_ridge_model = Ridge(alpha=best_lambda_ridge)
final_ridge_model.fit(X_train_std, y_train)

# Calcule o MSE no conjunto de validação
validation_mse = -cross_val_score(final_ridge_model, X_validation_std, y_validation, cv=3, scoring='neg_mean_squared_error')

# Calcule o desvio padrão do MSE no conjunto de validação
std_validation_mse = np.std(validation_mse)

# Plote a curva de validação para Ridge com barra de erro
plt.figure(figsize=(10, 5))
plt.plot(lambda_values, validation_scores_ridge, label='Ridge')
plt.scatter(best_lambda_ridge, validation_scores_ridge[np.where(lambda_values==best_lambda_ridge)[0][-1]],c='r', label='best lambda', zorder=3)
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.title('Curva de validação - Ridge')
plt.legend()
plt.grid()
plt.show()

print(f'Best Lambda for Ridge: {best_lambda_ridge}')