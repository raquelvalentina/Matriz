
import numpy as np
def detectar_listas_en_archivo(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        contenido = archivo.read()
        lineas = contenido.split('\n')
        index = len(lineas)
        matriz_diseno = []
        vector_de_observaciones = []
        vector_de_paramentrosX=[]

        for var in range(index):
            if 'vector de observaciones' in lineas[var]:
                elementos = lineas[var].split('=')[1].strip()[1:-1].split(', ')
                vector_de_observaciones = [elemento.strip("'") for elemento in elementos]
            elif 'matriz_diseno' in lineas[var] or '[]' in lineas[var]:
                elementos = lineas[var].split('=')[1].strip()[1:-1].split('], [')
                matriz_diseno = [elemento.strip("[]'").split("', '") for elemento in elementos]

            elif 'vector de parametros desconocidos' in lineas[var]:
                elementos = lineas[var].split('=')[1].strip()[1:-1].split('], [')
                vector_de_paramentrosX = [elemento.strip("[]'").split("', '") for elemento in elementos]


    return vector_de_observaciones, matriz_diseno, vector_de_paramentrosX

nombre_archivo1= ("archivo.txt")

vector_observaciones1, matriz_diseno1, vector_de_paramentrosX1 = detectar_listas_en_archivo(nombre_archivo1)

def gauss_markov(vector_observaciones, matriz_coeficientes, vector_parametros_desconocidos):
    # Convertir los valores de entrada en matrices numpy
    Y = np.array(vector_observaciones)
    A = np.array(matriz_coeficientes)
    X = np.array(vector_parametros_desconocidos)
    
    # Calcular la solución
    ATA = np.dot(A.T, A)
    ATY = np.dot(A.T, Y)
    X_hat = np.linalg.inv(ATA).dot(ATY)
    
    # Calcular los errores residuales
    e_hat = np.dot(A, X_hat) - Y
    
    # Calcular la precisión del ajuste
    n = len(Y)
    m = len(X_hat)
    sigma_hat_squared = np.dot(e_hat.T, e_hat) / (n - m)
    
    # Calcular la matriz de varianzas y covarianzas
    XX_hat = sigma_hat_squared * np.linalg.inv(ATA)
    
    # Imprimir los resultados
    for i in range(m):
        print(f"x_{i+1} = {X_hat[i]} ± {np.sqrt(XX_hat[i, i])}")

# Ejemplo de uso

gauss_markov(vector_observaciones1, matriz_diseno1, vector_de_paramentrosX1)


