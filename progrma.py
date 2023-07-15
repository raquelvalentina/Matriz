import numpy as np

# Paso 1: Leer el archivo de datos
nombre_archivo = input("Ingrese el nombre del archivo: ")
datos = np.load(nombre_archivo)

# Paso 2: Identificar la matriz de diseño
matriz_diseno = datos['matriz_diseno']

# Paso 3: Identificar el vector de observaciones
vector_observaciones = datos['vector_observaciones']

# Paso 4: Calcular el vector solución
vector_solucion = np.linalg.lstsq(matriz_diseno, vector_observaciones, rcond=None)[0]

# Paso 5: Calcular el vector de errores residuales
vector_errores_residuales = vector_observaciones - np.dot(matriz_diseno, vector_solucion)

# Paso 6: Calcular la precisión del ajuste
precision_ajuste = np.linalg.norm(vector_errores_residuales)

# Paso 7: Calcular la matriz de varianzas y covarianzas
matriz_varianzas_covarianzas = np.linalg.inv(np.dot(matriz_diseno.T, matriz_diseno))

# Paso 8: Imprimir el archivo
print("Archivo de datos:")
print(datos)

# Paso 9: Imprimir los resultados del proceso
print("Matriz de diseño:")
print(matriz_diseno)
print("Vector de observaciones:")
print(vector_observaciones)

# Paso 10: Crear un archivo llamado resultados
archivo_resultados = open("resultados.txt", "w")

# Paso 11: Agregar los resultados al archivo resultados
archivo_resultados.write("Vector solución:\n")
archivo_resultados.write(str(vector_solucion) + "\n")
archivo_resultados.write("Vector de errores residuales:\n")
archivo_resultados.write(str(vector_errores_residuales) + "\n")
archivo_resultados.write("Precisión del ajuste:\n")
archivo_resultados.write(str(precision_ajuste) + "\n")
archivo_resultados.write("Matriz de varianzas y covarianzas:\n")
archivo_resultados.write(str(matriz_varianzas_covarianzas) + "\n")

archivo_resultados.close()
'''Este código lee el nombre del archivo de datos proporcionado por el usuario, calcula el vector solución, el vector de errores residuales, la precisión del ajuste y la matriz de varianzas y covarianzas. Luego, imprime el archivo de datos y los resultados del proceso. Finalmente, crea un archivo llamado "resultados.txt" y agrega los resultados al archivo.'''