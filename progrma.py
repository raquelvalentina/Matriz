import numpy as np

def detectar_listas_en_archivo(nombre_archivo):
    vector_de_observaciones = []
    matriz_diseno = []

    with open(nombre_archivo, 'r') as archivo:
        contenido = archivo.read()
        lineas = contenido.split('\n')

        for linea in lineas:
            if 'vector de observaciones' in linea:
                elementos = linea.split('=')[1].strip()[1:-1].split(', ')
                vector_de_observaciones = [elemento.strip("'") for elemento in elementos]
            elif 'matriz_diseno' in linea:
                elementos = linea.split('=')[1].strip()[2:].split('], [')
                matriz_diseno = [elemento.strip("[]'").split("', '") for elemento in elementos]

    return vector_de_observaciones, matriz_diseno

nombre_archivo = input("Ingrese el nombre del archivo: ")

vector_observaciones, matriz_diseno = detectar_listas_en_archivo(nombre_archivo)

print("Matriz de diseño:")
print(matriz_diseno)
print("Vector de observaciones:")
print(vector_observaciones)