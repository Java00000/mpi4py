"""
Modul zum Lösen der Poissongleichung.
"""
import time
import numpy as np
# pylint: disable=E0611
from mpi4py import MPI
from cli_helper import PDEParameter
from pde_helper import init_matrices, get_display_matrix
from pde_lib import iterate


def main(cli_params: PDEParameter):
    """Die Hauptfunktion

    Args:
        cli_params (PDEParameter): Die Parameter aus der Kommandozeile
    """
    # MPI Vorbereiten
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    # Initialisieren der Matrizen zur Berechnung
    matrices = init_matrices(cli_params)

    # Hier beginnt die Berechnung
    start_time = time.time()
    if cli_params.is_method_jacobi():
        # Jacobi Verfahren seriell und parallel unterscheiden
        if world_size == 1:
            statistics = jacobi_solver(
                matrices, cli_params)
        else:
            statistics = jacobi_solver_mpi(
                matrices, cli_params)
    elif cli_params.is_method_gauss():
        # Gauss Verfahren seriell und parallel unterscheiden
        if world_size == 1:
            statistics = gauss_solver(
                matrices, cli_params)
        else:
            statistics = gauss_solver_mpi(
                matrices, cli_params)
    end_time = time.time()

    # Ausgabe von Statistiken und der Lösung durch Rang 0
    if rank == 0:
        statistics['duration'] = end_time - start_time
        print(f"Number of iterations: {statistics['iterations']}")
        print(f"Precision: {statistics['precision']}")
        print(f"Duration: {statistics['duration']}")

    # Zusammenstellen der Displaymatrix
    display_matrix = get_display_matrix(matrices[0], cli_params)

    # Rang 0 zeigt die Displaymatrix an
    if rank == 0:
        np.set_printoptions(precision=2, suppress=True)
        print(display_matrix)


def jacobi_solver_mpi(matrices, cli_params):
    """Paralleler PDE Solver mit dem Jacobi Verfahren

    Args:
        matrices (np.ndarray): Berechnungsmatrizen
        cli_params (PDEParameter): CLI Parameter

    Returns:
        dict: Ein dict mit statistischen Informationen
    """


    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    lower_neighbour = rank - 1
    upper_neighbour = rank + 1

    statistics = {'iterations': 0, 'precision': 0}
    matrix_a = 0
    matrix_b = 1

    termination = cli_params.termination

    # Setzen der Interationsanzahl
    if cli_params.is_termination_iter():
        iterations = cli_params.get_termination_value()
    else:
        # Iterations wird für Precision auf 1 gesetzt. Wir nutzen das Setzen auf 0 zum Abbruch
        iterations = 1

    while iterations > 0:
        strip_maxresiduum= iterate(matrices,matrix_a,matrix_b,termination,iterations)
        if lower_neighbour>=0:
            comm.Sendrecv(sendbuf=matrices[matrix_b][1],dest=lower_neighbour,
                          sendtag=1,recvbuf=matrices[matrix_b][0],
                          source=lower_neighbour,recvtag=0)
        if upper_neighbour<world_size:
            comm.Sendrecv(sendbuf=matrices[matrix_b][-2],dest=upper_neighbour,
                          sendtag=0,recvbuf=matrices[matrix_b][-1],
                          source=upper_neighbour,recvtag=1)
        matrix_a, matrix_b = matrix_b, matrix_a

        if cli_params.is_termination_prec():

            if strip_maxresiduum < cli_params.get_termination_value():
                #comm.Allreduce(strip_maxresiduum, MPI.MAX)
                iterations = 0
        else:
            iterations -= 1


        statistics['iterations'] += 1

    statistics['precision'] = strip_maxresiduum
    return statistics


def gauss_solver_mpi(matrices, cli_params):
    """Paralleler PDE Solver mit dem Gauss Verfahren

    Args:
        matrices (np.ndarray): Berechnungsmatrizen
        cli_params (PDEParameter): CLI Parameter

    Returns:
        dict: Ein dict mit statistischen Informationen
    """
    pass
    #TODO: Implementierung Gauss-Seidel Verfahren


def jacobi_solver(matrices, cli_params):
    """Serieller PDE Solver mit dem Jacobi Verfahren

    Args:
        matrices (np.ndarray): Die Berechnungsmatrizen
        cli_params (PDEParams): Die Parameter aus der CLI

    Returns:
        dict: Statistik
    """
    statistics = {'iterations': 0, 'precision': 0}
    matrix_a = 0
    matrix_b = 1

    termination = cli_params.termination

    # Setzen der Interationsanzahl
    if cli_params.is_termination_iter():
        iterations = cli_params.get_termination_value()
    else:
        # Iterations wird für Precision auf 1 gesetzt. Wir nutzen das Setzen auf 0 zum Abbruch
        iterations = 1

    while iterations > 0:
        maxresiduum = iterate(matrices, matrix_a, matrix_b, termination, iterations)

        # In der seriellen Version ist dies die einzige Zeile, die das Jacobi
        # vom Gauss-Seidel Verfahren unterscheidet. Hier werden die Matrizen gatauscht.
        matrix_a, matrix_b = matrix_b, matrix_a

        if cli_params.is_termination_prec():
            if maxresiduum < cli_params.get_termination_value():
                iterations = 0
        else:
            iterations -= 1
        statistics['iterations'] += 1
    statistics['precision'] = maxresiduum
    return statistics


def gauss_solver(matrices, cli_params):
    """Serieller PDE Solver mit dme Gauss Verfahren

    Args:
        matrices (np.ndarray): Die Berechnungsmatrizen
        disturbance_matrix (np.ndarray): Die Störmatrix
        cli_params (PDEParams): Die Parameter aus der CLI

    Returns:
        dict: Statistik
    """
    statistics = {'iterations': 0, 'precision': 0}
    matrix_a = 0
    matrix_b = 0

    termination = cli_params.termination

    # Setzen der Interationsanzahl
    if cli_params.is_termination_iter():
        iterations = cli_params.get_termination_value()
    else:
        # Iterations wird für Precision auf 1 gesetzt. Wir nutzen das Setzen auf 0 zum Abbruch
        iterations = 1

    while iterations > 0:
        maxresiduum = iterate(matrices, matrix_a, matrix_b, termination, iterations)

        if cli_params.is_termination_prec():
            if maxresiduum < cli_params.get_termination_value():
                iterations = 0
        else:
            iterations -= 1
        statistics['iterations'] += 1
    statistics['precision'] = maxresiduum
    return statistics


if __name__ == "__main__":
    params = PDEParameter()
    main(params)
