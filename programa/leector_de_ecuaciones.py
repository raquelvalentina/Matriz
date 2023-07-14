import PyPDF2
import regex as re
import pdfplumber
from sympy import Matrix, parse_expr, symbols
import cv2
import numpy as np
  

from sympy import Matrix
import numpy as np

def mostrar_valores_con_subindices(archivoPDF):
    valores = []
    caracteres_especiales = ['…', ':', '+', '=', ']', '[', 'y', 'a', 'x']
    
    with pdfplumber.open(archivoPDF) as pdf:
        primera_pagina = pdf.pages[0]  # Obtener la primera página del PDF
        texto = primera_pagina.extract_text()
        
        for caracter_especial in caracteres_especiales:
            texto = texto.replace(caracter_especial, '')

        lineas = texto.split('\n')
        linesS = [1, 3, 4, 7]
        valores_coeficiente = ['y', 'a', 'a', 'a', 'a', 'x'] 

        for lineS in linesS:
            valores_subindice = lineas[lineS].split()
            if len(valores_subindice) == len(valores_coeficiente):
                for coeficiente, subindice in zip(valores_coeficiente, valores_subindice):
                    subindice = np.array(subindice)
                    coeficiente_subindice = f'{coeficiente}{subindice}'
                    valores.append(coeficiente_subindice)

    return valores

archivo = './doc/programacion.pdf'
resultados = mostrar_valores_con_subindices(archivo)

# Crear una matriz con los elementos que contienen 'x'
matriz_x = np.array([elem for elem in resultados if 'x' in elem])

# Crear una matriz con los elementos que contienen 'a'
matriz_a = np.array([elem for elem in resultados if 'a' in elem])

# Crear una matriz con los elementos que no contienen ni 'x' ni 'a'
matriz_y = np.array([elem for elem in resultados if 'x' not in elem and 'a' not in elem])

# Crear una matriz simbólica con las matrices obtenidas
matriz_simbolica = Matrix([[matriz_x], [matriz_a], [matriz_y]])


print(matriz_simbolica)