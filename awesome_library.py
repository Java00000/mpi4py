"""
Bibliothek für Dummy Berechnungen.
"""
import numpy as np


def calculate_stuff(matrix: np.ndarray):
    """Platzhalter für eine Berechnung.
    Hier werden einfach alle Einträge der Matrix zufällig neu sortiert.
    An dieser Stelle ist das nur ein Platzhalter für eine echte Rechnung, wie
    wir sie z.B. beim Jacobi Verfahren ausführen werden.

    Args:
        matrix (np.ndarray): Die Matrix die geshuffelt werden soll.

    Returns:
        np.ndarray: Die neu geordnete Matrix
    """
    np.random.shuffle(matrix)
    return matrix


def initialize_matrix(start_row: int, end_row: int, problem_width: int) -> np.ndarray:
    """Inititalisiert eine Submatrix zwischen zwei Zeilen.

    Args:
        start_row (int): Startzeilenindex
        end_row (int): Endzeilenindex
        problem_width (int): Die Matrixbreite

    Returns:
        np.ndarray: Die Untermatrix, die für den Prozess erstellt wird.
    """
    number_of_rows = end_row - start_row
    return np.random.rand(number_of_rows, problem_width)
