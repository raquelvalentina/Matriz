import tabula
import numpy as np

def extraer_ecuaciones_y_matrices(pdf_file):
    ecuaciones = []
    matrices = []

    tablas = tabula.read_pdf(pdf_file, pages='all')
    for table in tablas:
        print(f'tabla:\n{table}\n\n\n')
        valores = []
        for row in table:
            fila = []
            fila.append(row.strip())
            for cell in row:
              columna = []
              columna.append(row.strip())
            valores.append(columna)
            valores.append(fila)
        
        matriz_numpy = np.array(valores)
        matrices.append(matriz_numpy)

    return ecuaciones, matrices
# Ejemplo de uso
documento = "./programa//doc/programacion.pdf"
ecuaciones_extraidas, matrices_extraidas = extraer_ecuaciones_y_matrices(documento)

print(matrices_extraidas)
print(ecuaciones_extraidas)