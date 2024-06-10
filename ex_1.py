import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def fitar_curva(x, t, model):
    X = np.linspace(0,1,100)
    # regressão
    poly = PolynomialFeatures(degree=9) 
    A = poly.fit_transform(x.reshape(-1, 1)) # calculo da matriz A
    model.fit(A, t) # treinamento
    A = poly.fit_transform(X.reshape(-1, 1))
    y=model.predict(A) # predição

    return y, model.coef_

def plotar_curva(x,y,modelo):
    # função geradora
    X = np.linspace(0,1,100)
    modelo_gerador = np.sin(2*np.pi*X)

    # resultados
    plt.figure()
    plt.plot(X,modelo_gerador,color='green', label='Modelo Gerador')
    plt.scatter(x, t, facecolors='none', edgecolors="blue", label ='Dados de treinamento')
    plt.plot(X,y,color='red', label = modelo)
    plt.title(f'Solução para {modelo}')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend()
    plt.show()

def plotar_tudo(x,y_list,modelos,N):
    # função geradora
    X = np.linspace(0,1,100)
    modelo_gerador = np.sin(2*np.pi*X)

    # resultados
    plt.figure()
    plt.plot(X,modelo_gerador,color='green', label='Modelo Gerador')
    plt.scatter(x, t, facecolors='none', edgecolors="blue", label ='Dados de treinamento')
    color_list = ["k","b", "r"]
    for i, y in enumerate(y_list):
        plt.plot(X,y,color=color_list[i],label=modelos[i])
        # plt.plot(X,y,label=modelos[i])
    plt.title(f'Solução para N = {N}')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend()
    plt.show()

def listar_coefs(coef_list):
    model_names = ['LS', 'Ridge', 'Lasso']

    coef_dict = {'Modelo': model_names}
    for i in range(10):  # assumindo que existem 10 coeficientes
        coef_dict[f'w{i}'] = [f'{coef[i]:.2f}' for coef in coef_list]

    coef_table = pd.DataFrame(coef_dict)
    print(coef_table.to_latex(index=False))
    
if __name__=="__main__":
    # amostra
    np.random.seed(42) # congelando a seed para gerar os mesmos dados de treinamento para todos os itens 
    N = 50 # tamanho da amostra
    x = np.linspace(0,1,N)
    t = np.sin(2*np.pi*x) + np.random.normal(0, 0.5, size=N)
    # curve fitting
    lambda_ridge = 1e-3
    lambda_lasso = 1e-4
    modelos = [LinearRegression(), Ridge(alpha=lambda_ridge), Lasso(alpha=lambda_lasso)]
    # modelos = [Ridge(alpha=1e-1),Ridge(alpha=1e-2),Ridge(alpha=1e-3),Ridge(alpha=1e-4),Ridge(alpha=1e-5),Ridge(alpha=1e-6)]
    # modelos = [Lasso(alpha=1e-2),Lasso(alpha=1e-3),Lasso(alpha=1e-4),Lasso(alpha=1e-5),Lasso(alpha=1e-6),Lasso(alpha=1e-7)]
    y_list = list()
    w_list = list()
    for model in modelos:
        y, w = fitar_curva(x, t, model)
        y_list.append(y)
        w_list.append(w)
        # plotar_curva(x,y,model)

    listar_coefs(w_list)
    plotar_tudo(x,y_list,modelos,N)
    
    