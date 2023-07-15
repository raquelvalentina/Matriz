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


def gauss_markov(y, A):
    n = len(y)
    m = len(A[0])
    
    # Crear matriz de diseño A
    A = np.array(A)
    
    # Crear vector de observaciones y
    y = np.array(y)
    
    # Calcular la matriz de pesos P
    P = np.eye(n)
    
    # Calcular la matriz de coeficientes A transpuesta
    A_transpose = np.transpose(A)
    
    # Calcular el producto A transpuesta * P * A
    product = np.dot(A_transpose, np.dot(P, A))
    
    # Calcular la inversa de A transpuesta * P * A
    inverse = np.linalg.inv(product)
    
    # Calcular el producto A transpuesta * P * y
    product2 = np.dot(A_transpose, np.dot(P, y))
    
    # Calcular el vector solución X
    X_hat = np.dot(inverse, product2)
    
    # Calcular el vector de errores residuales e
    e_hat = np.dot(A, X_hat) - y
    
    # Calcular la precisión del ajuste σ^2
    sigma_hat_squared = np.dot(np.transpose(e_hat), np.dot(P, e_hat)) / (n - m)
    
    # Calcular la matriz de varianzas y covarianzas ΣXX
    Sigma_XX = sigma_hat_squared * inverse
    
    return X_hat, e_hat, sigma_hat_squared, Sigma_XX

# Ejemplo de uso
# y = vector_observaciones1
# A = matriz_diseno1

# X_hat, e_hat, sigma_hat_squared, Sigma_XX = gauss_markov(y, A)

# print("Vector solución X_hat:")
# print(X_hat)

# print("Vector de errores residuales e_hat:")
# print(e_hat)

# print("Precisión del ajuste sigma_hat_squared:")
# print(sigma_hat_squared)

# print("Matriz de varianzas y covarianzas Sigma_XX:")
# print(Sigma_XX)



# print("Matriz de diseño:")
# print(matriz_diseno1)
# print("Vector de observaciones:")
# print(vector_observaciones1)