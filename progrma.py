import numpy as np

def calcular_valores(vector_de_observaciones, matriz_diseno, A, X, e, sigma_o2):
    # Cálculo de la matriz A
    A = matriz_diseno.T

    # Cálculo de la matriz (A^T * P * A)^-1
    ATA_inv = np.linalg.inv(np.dot(np.dot(A.T, sigma_o2), A))

    # Cálculo de X_hat
    X_hat = np.dot(ATA_inv, np.dot(A.T, vector_de_observaciones))

    # Cálculo del vector de residuos e_hat
    e_hat = vector_de_observaciones - np.dot(A, X_hat)

    # Cálculo de sigma_hat^2
    sigma_hat_squared = np.dot(np.dot(e_hat.T, sigma_o2), e_hat) / (len(vector_de_observaciones) - len(X_hat))

    # Cálculo de MATRIZ_DE_VARIANZA_COVARIANZA
    MATRIZ_DE_VARIANZA_COVARIANZA = ATA_inv * sigma_hat_squared

    return X_hat, e_hat, sigma_hat_squared, MATRIZ_DE_VARIANZA_COVARIANZA

# Datos proporcionados
vector_de_observaciones = np.array([103.758, 104.113, 105.713, 107.313, 108.913, 110.513, 112.113])
matriz_diseno = np.array([[1695.557, 700.000, 720.000, 740.000, 760.000, 780.000, 800.000], [1, 1, 1, 1, 1, 1, 1]])
A = np.array([[0.000104815, -0.07779615], [-0.07779615, 57.884906]])
X = np.array([0.079997848, 48.11466023])
e = np.array([0.0003, -0.0002, -0.0001, -0.0001, 0.0000, 0.0000, 0.0001])
sigma_o2 = np.array([[2.43507E-08, 2.55232E-12, -1.8944E-09], [0, 1.8944E-09, 1.8944E-09]])

# Cálculo de los valores
X_hat, e_hat, sigma_hat_squared, MATRIZ_DE_VARIANZA_COVARIANZA = calcular_valores(vector_de_observaciones, matriz_diseno, A, X, e, sigma_o2)

print("X_hat:", X_hat)
print("VECTOR_ERRORES_RESIDUALES:", e_hat)
print("PRECISION_DE_AJUSTE:", sigma_hat_squared)
print("MATRIZ_DE_VARIANZA_COVARIANZA:", MATRIZ_DE_VARIANZA_COVARIANZA)
