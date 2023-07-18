import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np
import os
from io import StringIO

from main import resolver_sistema


class TestResolverSistema(unittest.TestCase):
    @patch("pandas.read_csv")
    def test_solucion_conocida(self, mock_read_csv):
        # Crear datos de prueba
        X = pd.DataFrame({
            "x1": [1, 2, 3, 4, 5, 13.00, 7],
            "x2": [8, 9, 10, 11, 12, 13, 14]
        })
        Y = pd.DataFrame({
            "y": [7, 8, 9, 10, 11, 12, 13]
        })

        # Configurar el mock para devolver los datos de prueba
        mock_read_csv.side_effect = [X, Y]

        # Calcular la solución esperada
        beta_hat_esperado = np.linalg.inv(X.T @ X) @ X.T @ Y

        # Ejecutar la función
        resultados_beta, _, _ = resolver_sistema("X.csv", "Y.csv", "resultados.csv")

        # Comprobar que la solución es correcta
        np.testing.assert_array_almost_equal(resultados_beta["beta_hat"].values, beta_hat_esperado.values.ravel())

    @patch("pandas.read_csv")
    def test_datos_aleatorios(self, mock_read_csv):
        # Crear datos aleatorios
        X = pd.DataFrame(np.random.rand(100, 3))
        Y = pd.DataFrame(np.random.rand(100, 1))

        # Configurar el mock para devolver los datos de prueba
        mock_read_csv.side_effect = [X, Y]

        # Ejecutar la función y comprobar que no hay errores
        try:
            resolver_sistema("X.csv", "Y.csv", "resultados.csv")
        except Exception as e:
            self.fail(f"resolver_sistema raised Exception with random data: {e}")

    @patch("pandas.read_csv")
    def test_archivo_no_existente(self, mock_read_csv):
        # Configurar el mock para lanzar un error de archivo no encontrado
        mock_read_csv.side_effect = FileNotFoundError

        # Ejecutar la función y comprobar que se lanza un FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            resolver_sistema("X.csv", "Y.csv", "resultados.csv")

    @patch("pandas.read_csv")
    def test_datos_insuficientes(self, mock_read_csv):
        # Crear datos insuficientes (una única fila)
        X = pd.DataFrame({"x1": [1]})
        Y = pd.DataFrame({"y": [2]})

        # Configurar el mock para devolver los datos de prueba
        mock_read_csv.side_effect = [X, Y]

        # Ejecutar la función y comprobar que se lanza un ValueError
        with self.assertRaises(ValueError):
            resolver_sistema("X.csv", "Y.csv", "resultados.csv")

    @patch("pandas.read_csv")
    def test_multicolinealidad(self, mock_read_csv):
        # Crear datos con multicolinealidad perfecta
        X = pd.DataFrame({
            "x1": [1, 2, 3],
            "x2": [1, 2, 3]
        })
        Y = pd.DataFrame({"y": [4, 5, 6]})

        # Configurar el mock para devolver los datos de prueba
        mock_read_csv.side_effect = [X, Y]

        # Ejecutar la función y comprobar que se lanza un LinAlgError
        with self.assertRaises(np.linalg.LinAlgError):
            resolver_sistema("X.csv", "Y.csv", "resultados.csv")


if __name__ == "__main__":
    unittest.main()
