import pandas as pd
import numpy as np


def resolver_sistema(nombre_archivo_X, nombre_archivo_Y, nombre_archivo_resultados):
    # Leer los datos
    X = pd.read_csv(nombre_archivo_X).values
    Y = pd.read_csv(nombre_archivo_Y).values

    # Asegurarse de que Y es un vector columna
    if Y.shape[1] > 1:
        Y = Y.reshape(-1, 1)

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

    # Crear un DataFrame con los resultados de beta_hat
    resultados_beta = pd.DataFrame({
        "beta_hat": beta_hat.flatten(),
    })

    resultados_beta["r2"] = r2

    # Crear un DataFrame con los errores
    resultados_errores = pd.DataFrame({
        "errores": errores.flatten(),
    })

    # Crear un DataFrame con la varianza y covarianza
    resultados_var_covar = pd.DataFrame({
        "var_covar_diagonal": np.diag(var_covar)
    })

    # Guardar los resultados en diferentes archivos CSV
    resultados_beta.to_csv(nombre_archivo_resultados + "_beta.csv", index=False)
    resultados_errores.to_csv(nombre_archivo_resultados + "_errores.csv", index=False)
    resultados_var_covar.to_csv(nombre_archivo_resultados + "_var_covar.csv", index=False)

    print(f"Los resultados se han guardado en {nombre_archivo_resultados}_beta.csv, {nombre_archivo_resultados}_errores.csv, y {nombre_archivo_resultados}_var_covar.csv.")

    return resultados_beta, resultados_errores, resultados_var_covar
