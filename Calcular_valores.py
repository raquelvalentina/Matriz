import numpy as np
import sympy as sp

def detectar_listas_en_archivo(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        contenido = archivo.read()
        lineas = contenido.split('\n')
        index = len(lineas)
        matriz_diseno = []
        vector_de_observaciones = []
        vector_de_parametrosX = []

        for var in range(index):
            if 'vector_de_observaciones' in lineas[var]:
                elementos = lineas[var].split('=')[1].strip()[1:-1].split(' ')
                elementos = [elemento.rstrip(',') for elemento in elementos]
                vector_de_observaciones = [float(elemento) for elemento in elementos]
            elif 'matriz_diseno' in lineas[var]:
                elementos = lineas[var].split('=')[1].strip()[2:-2].split('], [')
                valores = [elemento.split(', ') for elemento in elementos]
                matriz_diseno = [[float(valor) if valor != '' else 0.0 for valor in fila] for fila in valores]
            elif 'vector_de_parametrosX' in lineas[var]:
                elementos = lineas[var].split('=')[1].strip()[2:-2].split('], [')
                valores = [elemento.split(', ') for elemento in elementos]
                vector_de_parametrosX = [[float(valor) if valor != '' else 0.0 for valor in fila] for fila in valores]

    return vector_de_observaciones, matriz_diseno, vector_de_parametrosX

nombre_archivo1 = 'archivo.txt'
vector_observaciones1, matriz_diseno1, vector_de_parametrosX1 = detectar_listas_en_archivo(nombre_archivo1)

def gauss_markov(y, a):
    resultados = [] 
    Y = sp.Matrix(y)
    A = sp.Matrix(a)
    P = A.T * A

    X_hat = (A.T * P * A).inv() * A.T * P * Y

    e_hat = A * X_hat - Y

    n, m = A.shape
    sigma_hat_squared = e_hat.T * P * e_hat / (n - m)

    X_X = sigma_hat_squared * (A * P * A.T).inv() 

    for i in range(len(X_hat)):
        resultados.append(f'resultados_{i+1}: {X_hat[i]} +- {sp.sqrt(X_X[i, i])}')

    resultados_dict = {
        'VECTOR_ERRORES_RESIDUALES': e_hat,
        'PRECISION_DE_AJUSTE': sigma_hat_squared,
        'MATRIZ_DE_VARIANZA_COVARIANZA': X_X
    }

    resultados.append(resultados_dict)

    return resultados

vector_observaciones1= [1, 2, 3, 5]

matriz_diseno1 = [[1, 2, 3, 4], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

if __name__ == '__main__':

    l = gauss_markov(vector_observaciones1, matriz_diseno1)

    for resultado in l:
        print(resultado)
    with open('resultados.txt', 'w') as resultados:
        for resultado in l:
            if isinstance(resultado, dict):
                resultados.write("VECTOR_ERRORES_RESIDUALES:\n")
                resultados.write(str(resultado['VECTOR_ERRORES_RESIDUALES']) + '\n')
                resultados.write("PRECISION_DE_AJUSTE:\n")
                resultados.write(str(resultado['PRECISION_DE_AJUSTE']) + '\n')
                resultados.write("MATRIZ_DE_VARIANZA_COVARIANZA:\n")
                resultados.write(str(resultado['MATRIZ_DE_VARIANZA_COVARIANZA']) + '\n')
            else:
                resultados.write(resultado + '\n')


