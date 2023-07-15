import openai


openai.api_key = "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"


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


archivo = "./programa/doc/archivo.txt"

with open(archivo, "r") as f:
    texto = f.read()

prompt = f"""Siga las siguientes instrucciones \
 
Escriba un programa en python que pregunte al usuario el nombre del archivo donde se proporciona la matriz de diseño y el vector de observaciones del sistema de ecuaciones lineales por minimos cuadros el modelo de Gauss-Markov de ecuaciones de observacion. \
El codigo debe cumplir los pasos siguientes \

1- El programa debe leer el archivo de datos\
2- Identificar en el archivo la matriz de diseño (los cuales estan dentro de una lista)\
3- Identificar en el archivo el vector de observaciones (los cuales estan dentro de una lista)\
4- calcular vector solucion\
5- calcular vector de errores reiduales\
6- calcular la precision del ajuste\
7- calcular la matriz de varianzas y covarianzas\
8- Imprimir el archivo\
9- Imprimir los resultados del proceso indicando la matriz de diseño y el vector de observaciones \
10- Crea un archivo llamado resultados\
11- Agregar al archivo resultados los resultados de: [[VECTOR SOLUCION], [VECTOR DE ERRORES RESIDUALES], [PRECISION DEL AJUSTE], [MATRIZ DE VARIANZAS Y COVARIANZAS]]

proporcione el codigo en formato XML \

"""

response = get_completion(prompt)

print(response)
