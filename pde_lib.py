"""
Dieses Modul enthält Funktionen zum Lösen von PDEs mit dem Jacobi und Gauss-Seidel Verfahren.
"""
import numpy as np
from numba.pycc import CC
from cli_helper import TERMINATION_PREC

cc = CC('pde_lib')


@cc.export('iterate', 'f8(f8[:,:,:],i8,i8,i8,i8)')
def iterate(matrices: np.ndarray,
            matrix_in: int,
            matrix_out: int,
            termination: int,
            iteration: int) -> float:
    """Hier findet die eigentliche Berechnung statt. Die Matrix wird dabei einmal iteriert.

    Für das Jacobiverfahren werden matrix_in und matrix_out auf
    unterschiedliche Werte gesetzt, da es ein Gesamtschrittverfahren ist.
    Für das Gauss-Seidel Verfahren wird matrix_in = matrix_out gesetzt,
    da die Matrix während der Rechnung gleichzeitig in unterschiedlichen
    Iterationen ist.

    Args:
        matrices (np.ndarray): Die Berechnungsmatrizen
        matrix_in (int): Die Matrix die zur letzten Iteration gehört (lesen)
        matrix_out (int): Die Matrix die zur neuen Iteration gehört (schreiben)
        termination (int): Die Abbruchbedingung
        iteration (int): Die aktuelle Iteration

    Returns:
        float: Das maxresiduum für die Berechnungsmatrix nach der Iteration,
               wenn eine Abbruchbedingung erreicht wurde, sonst 0.
    """
    maxresiduum = 0
    dim_x, dim_y = matrices[matrix_in].shape
    for idx in range(1, dim_x - 1):
        for idy in range(1, dim_y - 1):
            star = 0.25 * (matrices[matrix_in, idx - 1, idy] +
                           matrices[matrix_in, idx + 1, idy] +
                           matrices[matrix_in, idx, idy - 1] +
                           matrices[matrix_in, idx, idy + 1])

            if termination == TERMINATION_PREC or iteration == 1:
                residuum = np.abs(matrices[matrix_in, idx, idy] - star)
                maxresiduum = residuum if residuum > maxresiduum else maxresiduum

            matrices[matrix_out, idx, idy] = star
    return maxresiduum


if __name__ == "__main__":
    cc.compile()
