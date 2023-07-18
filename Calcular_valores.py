import pandas as pd
import numpy as np

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

    return vector_de_observaciones, matriz_diseno

archivo= input('Indique el nombre del archivo:  ')

vector, matriz_de_coeficientes = detectar_listas_en_archivo(archivo)

def resolver_sistema(X_list, Y_list, nombre_archivo_resultados):
    # Leer los datos
    X = np.array(X_list)
    Y = np.array(Y_list)

    # Calcular el vector de parámetros
    XTX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XTX_inv @ X.T @ Y

    # Calcular el vector de errores residuales
    errores = Y - X @ beta_hat

    # Calcular la precisión del ajuste (R^2)
    ss_res = np.sum(errores ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Calcular la matriz de varianzas y covarianzas
    sigma2 = ss_res / (X.shape[0] - X.shape[1])
    var_covar = sigma2 * XTX_inv

    # Crear un DataFrame con los resultados de el vector de parámetros
    resultados_beta = pd.DataFrame({
        "beta_hat": beta_hat.flatten(),
    })

    resultados_beta["r2"] = r2

    # Crear un DataFrame con los errores residuales
    resultados_errores = pd.DataFrame({
        "errores": errores.flatten(),
    })

    # Crear un DataFrame con la matriz de varianza y covarianza
    resultados_var_covar = pd.DataFrame({
        "var_covar_diagonal": np.diag(var_covar)
    })

    # Guardar los resultados en diferentes archivos CSV
    resultados_beta.to_csv(nombre_archivo_resultados + "_beta.csv", index=False)
    resultados_errores.to_csv(nombre_archivo_resultados + "_errores_residuales.csv", index=False)
    resultados_var_covar.to_csv(nombre_archivo_resultados + "_matriz_de_varianza__covarianza.csv", index=False)

    print(f"Los resultados se han guardado en {nombre_archivo_resultados}_beta.csv, {nombre_archivo_resultados}_errores.csv, y {nombre_archivo_resultados}_var_covar.csv.")

    return resultados_beta, resultados_errores, resultados_var_covar

resolver_sistema(matriz_de_coeficientes, vector, 'resultados.txt')

