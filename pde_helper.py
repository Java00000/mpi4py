"""Dieses Modul stellt einige Hilfsfunktionen bereit, die für die
parallele Version des Poisson-Lösers hilfreich sein könnten.
"""

import sys
import numpy as np
# pylint: disable=E0611
from mpi4py import MPI
from cli_helper import PDEParameter


def _get_start_row(rank: int, world_size: int, matrix_size: int) -> tuple[int, int]:
    """Berechnet die Startzeile für den gegebene Rang

    Args:
        rank (int): Rang
        world_size (int): Weltgröße bzw. Gesamtanzahl der Ränge
        matrix_size (int): Die Größe der Matrix

    Returns:
        tuple[int, int]: Der erste Wert ist ohne Ghostlayer der zweite mit.
    """
    start_row = (rank * matrix_size) // world_size
    if not start_row == 0:
        return start_row, start_row - 1
    return start_row, start_row


def _get_end_row(rank: int, world_size: int, matrix_size: int) -> tuple[int, int]:
    """Berechnet die Endzeile für den gegebene Rang

    Args:
        rank (int): Rang
        world_size (int): Weltgröße bzw. Gesamtanzahl der Ränge
        matrix_size (int): Die Größe der Matrix

    Returns:
        tuple[int, int]: Der erste Wert ist ohne Ghostlayer der zweite mit.
    """
    end_row = ((rank + 1) * matrix_size) // world_size
    if not end_row == matrix_size:
        return end_row, end_row + 1
    return end_row, end_row


def _get_print_idxs(matrix_size: int, display_matrix_size: int = 11) -> np.ndarray:
    """Erzeugt eine Liste mit Indizes der Reihen bzw. Zeilen einer Matrix,
    welche für die Ausgabe in Frage kommen.

    Args:
        matrix_size (int): Die Größe der Matrix
        display_matrix_size (int, optional): Die Größe der Anzeigematrix. Standardwert 11.

    Returns:
        np.ndarray: Ein Vektor mit den Indizes.
    """
    return np.int32(np.round(
        np.linspace(0, matrix_size - 1, display_matrix_size, endpoint=True)))
    # Achtung ungleich: np.linspace(0, matrix_size - 1, display_matrix_size,
    #                               endpoint=True, dtype=np.int32)


# pylint: disable=R0914
def get_display_matrix(matrix: np.ndarray,
                       cli_params: PDEParameter,
                       display_matrix_size: int = 11) -> np.ndarray:
    """Erzeugt eine Matrix zum Anzeigen der Ergebnis.

    Args:
        matrix (np.ndarray): Die Matrix oder bei MPI Programmen der Matrixanteil
        cli_params (PDEParameter): Die CLI Parameter
        display_matrix_size (int, optional): Die Größe der Matrix. Defaults to 11.

    Returns:
        np.ndarray: Rang 0 bekommt die Anzeige Matrix, alle anderen Ränge Null.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    print_idxs = _get_print_idxs(cli_params.matrix_size, display_matrix_size)

    # Serieller Fall
    if world_size == 1:
        dim1_selection, dim2_selection = np.meshgrid(print_idxs, print_idxs)
        return matrix[dim1_selection, dim2_selection]

    # Paralleler Fall
    # Hier wird Gatherv benutzt, sodass zu keinem Zeitpunkt die gesamte Matrix
    # im Speicher eines einzelnen Prozess liegen muss.
    start_row, _ = _get_start_row(rank, world_size, cli_params.matrix_size)

    dim1_ranks = np.empty(display_matrix_size, dtype=np.int64)
    rdx = 0
    for row in print_idxs:
        for other_rank in range(0, world_size):
            other_start_row, _ = _get_start_row(
                other_rank, world_size, cli_params.matrix_size)
            other_end_row, _ = _get_end_row(
                other_rank, world_size, cli_params.matrix_size)
            if other_start_row <= row < other_end_row:
                dim1_ranks[rdx] = other_rank
                rdx += 1

    _, distribution_vector = np.unique(dim1_ranks, return_counts=True)
    local_print_idxs = print_idxs[dim1_ranks == rank] - max(0, start_row - 1)

    dim1_selection, dim2_selection = np.meshgrid(local_print_idxs, print_idxs)
    local_display_matrix = matrix[dim1_selection, dim2_selection]

    display_matrix = None
    if rank == 0:
        display_matrix = np.empty(
            (display_matrix_size, display_matrix_size), dtype=np.float64)

    comm.Gatherv(local_display_matrix.flatten('F'),
                 (display_matrix, distribution_vector * display_matrix_size), root=0)

    return display_matrix


def _init_calculation_matrices(cli_params: PDEParameter,
                               start_row: int,
                               end_row: int):
    """Inititalisiert die Matrizen zum Berechnen der Lösung.

    Args:
        cli_params (PDEParameter): Die Parameter aus der Kommandozeile
        start_row (int): Index der Startzeile
        end_row (int): Index der Endzeile

    Returns:
        np.ndarray: Ein 3d Array, mit einer oder zwei Matrizen.
    """
    num_rows = end_row - start_row

    number_of_matrices = 1 if cli_params.is_method_gauss() else 2
    matrices = np.zeros((number_of_matrices, num_rows,
                        cli_params.matrix_size), dtype=np.float64)

    slope = 1.0 / (cli_params.matrix_size - 1)
    for m_idx in range(number_of_matrices):
        if start_row == 0:
            matrices[m_idx, 0, :] = np.linspace(1., 0.,
                                                cli_params.matrix_size)
        if end_row == cli_params.matrix_size:
            matrices[m_idx, -1, :] = np.linspace(0., 1.,
                                                 cli_params.matrix_size)
        matrices[m_idx, :, 0] = np.linspace(-start_row * slope + 1.,
                                            -(end_row-1) * slope + 1.,
                                            num_rows)
        matrices[m_idx, :, -1] = np.linspace(start_row * slope,
                                             (end_row - 1) * slope,
                                             num_rows)
    return matrices


def init_matrices(cli_params: PDEParameter) -> np.ndarray:
    """Inititalisiert die Berechnungs und die Störmatrix.

    Die Berechnungsmatrizen werden je nach Verfahren unterschiedlich erstellt.
    Für das Jacobiverfahren werden zwei gleiche Matrizen angelegt, die in dem
    calclation_matrix array durch den ersten Index unterschieden werden können.
    Für das Gauss-Seidel Verfahren wird nur eine Matrix verwendet. Dennoch wird
    ein 3d Array angelegt, damit viele Rechenoperatioenen wiederverwendet werden
    können.

    Args:
        cli_params (PDEParameter): die Parameter aus der Kommandozeile

    Returns:
        np.ndarray: Berechnungsmatrix
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    _, start_row = _get_start_row(rank, world_size, cli_params.matrix_size)
    _, end_row = _get_end_row(rank, world_size, cli_params.matrix_size)

    num_rows = end_row - start_row

    # Wenn nicht jeder Rang mindestens 3 Zeilen hat, dann kann nicht gerechnet werden.
    # Aus diesem Grund brechen wir ab, wenn das nicht der Fall ist.
    stop = np.empty(1, dtype=bool)
    comm.Allreduce(np.array(num_rows < 3, dtype=bool), stop, MPI.LOR)
    if stop:
        if rank == 0:
            print(
                "The number of ranks is too large for the porblem size. \
                    Please restart with less ranks or increase the problem size.")
        sys.exit()

    calculation_matrix = _init_calculation_matrices(
        cli_params, start_row, end_row)
    return calculation_matrix
