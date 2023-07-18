import numpy as np

A = np.array([[1695.557, 700.000, 720.000, 740.000, 760.000, 780.000, 800.000], [1, 1, 1, 1, 1, 1, 1]])
P = np.array([[2.43507E-08, 2.55232E-12, -1.8944E-09], [0, 1.8944E-09, 1.8944E-09], [0, 0, 1.8944E-09]])
y = np.array([103.758, 104.113, 105.713, 107.313, 108.913, 110.513, 112.113])

# # Reshape y to make it a column vector (7x1)
# y_reshaped = y.reshape(-1, 1)

# # Reshape A to make it a 7x2 matrix
# A_reshaped = np.transpose(A)


# Perform the matrix operations
ATA_inv = np.linalg.inv(np.transpose(A) @ P @ A)
ATy = np.transpose(A) @ P @ y.reshape(-1, 1)
VECTOR_SOLUCION = ATA_inv @ ATy

print("VECTOR_SOLUCION:")
print(VECTOR_SOLUCION)

VECTOR_ERRORES_RESIDUALES = y.reshape(-1, 1) - A @ VECTOR_SOLUCION

print("VECTOR_ERRORES_RESIDUALES:")
print(VECTOR_ERRORES_RESIDUALES)

PRECISION_DE_AJUSTE = 1 - (np.transpose(VECTOR_ERRORES_RESIDUALES) @ P @ VECTOR_ERRORES_RESIDUALES) / (np.transpose(y.reshape(-1, 1)) @ P @ y.reshape(-1, 1))

print("PRECISION_DE_AJUSTE:")
print(PRECISION_DE_AJUSTE)

MATRIZ_DE_VARIANZA_COVARIANZA = ATA_inv @ P

print("MATRIZ_DE_VARIANZA_COVARIANZA:")
print(MATRIZ_DE_VARIANZA_COVARIANZA)
