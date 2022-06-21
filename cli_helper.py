"""
Modul zum laden und verwalten der CLI Parameter für die PDE Programme.
"""
import argparse
from typing import Any, Union

METHOD_JACOBI = 0
METHOD_GAUSS = 1
TERMINATION_ITER = 0
TERMINATION_PREC = 1


class PDEParameter:
    """Objekte dieser Klasse erlauben einen einfachen Zugriff auf die Benutzerparmeter.
    """

    def __init__(self):
        """init
        """
        super().__init__()
        self.parser = _setup_parser()
        self._read_parameters()

    def _read_parameters(self):
        """Liest die Parameter von der Kommandozeile ein.
        """
        args = self.parser.parse_args()

        self.matrix_size = args.system_size

        self.method = METHOD_JACOBI if args.method == 'jacobi' else METHOD_GAUSS
        self.termination = TERMINATION_ITER \
            if args.precision is None else TERMINATION_PREC
        self.termination_value = args.iterations \
            if self.termination == TERMINATION_ITER else args.precision
        self.savefile = args.savefile

    def get_parser(self) -> argparse.ArgumentParser:
        """Gibt den evaluierten ArgumentParser zurück.

        Returns:
            argparse.ArgumentParser: Der Argumentparser.
        """
        return self.parser

    def is_method_jacobi(self) -> bool:
        """Prüft auf die Jacobimethode

        Returns:
            bool: True, wenn die gewählte Methode Jacobi ist.
        """
        return self.method == METHOD_JACOBI

    def is_method_gauss(self) -> bool:
        """Prüft auf die Gauss-Seidel-Methode.

        Returns:
            bool: True, wenn die gewählte Methode Gauss-Seidel ist.
        """
        return self.method == METHOD_GAUSS

    def is_termination_iter(self) -> bool:
        """Prüft ob nach einer bestimmten Anzahl von Iterationen abgebrochen werden soll.

        Returns:
            bool: True, wenn die Abbruchbedingung Iter ist.
        """
        return self.termination == TERMINATION_ITER

    def is_termination_prec(self) -> bool:
        """Prüft ob bei einer bestimmten Genauigkeit abgebrochen werden soll.

        Returns:
            bool: True, wenn bei einer bestimmten Genauigkeit abgebrochen werden soll.
        """
        return self.termination == TERMINATION_PREC

    def get_termination_value(self) -> Union[float, int]:
        """Gibt die Anazhl der Iterationen oder die Genauigkeit zurück,
           ab der abgebrochen werden soll.

        Returns:
            float, int: Die Anzahl der Iterationen oder die Genauigkeit
        """
        return self.termination_value


def check_system_size(value: Any) -> int:
    """Prüft auf eine ganzzahlige Zahl grüßer 10.

    Args:
        value (any): der zu überprüfende Wert

    Raises:
        argparse.ArgumentTypeError: Wenn die Zahl nicht größer 10 ist.

    Returns:
        int: Der Wert
    """
    value = check_positive_int(value)
    if value < 11:
        raise argparse.ArgumentTypeError("{} is too small for system size.\
                                         11 is minimum.".format(value))
    return value


def check_positive(value: Any) -> Any:
    """Prüft ob ein Wert positiv ist.

    Args:
        value (any): der zu überprüfende Wert.

    Raises:
        argparse.ArgumentTypeError: Wenn der Wert nicht positiv ist.

    Returns:
        float: Der Wert
    """
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError(
            "{} is an invalid positive value".format(value))
    return value


def check_positive_int(value: Any) -> int:
    """Prüft ob ein Wert ganzzahlig und positiv ist.

    Args:
        value (any): Der zu überprüfende Wert.

    Raises:
        argparse.ArgumentTypeError: Wenn der Wert nicht ganzzahlig und positiv ist.

    Returns:
        int: Der Wert als Integer.
    """
    # try to convert to int
    value = check_positive(value)
    try:
        value = int(value)
    except ValueError as val_err:
        raise argparse.ArgumentTypeError(
            "Value could not be converted to an integer value.") from val_err
    return value


def _setup_parser() -> argparse.ArgumentParser:
    """Erzeugt den parser.

    Returns:
        argparse.ArgumentParser: Der ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Programm to solve partial\
                                    differential equations.')
    parser.add_argument('system_size', type=check_system_size,
                        help='The system size. Has to be 10 or larger.')
    parser.add_argument('method', type=str, choices=['jacobi', 'gauss'],
                        help='The method to solve the system.')
    parser.add_argument('-s', '--savefile', type=str, default= None,
                        help='Path to the hdf5 file to save the solution maitrx.')

    stop_group = parser.add_mutually_exclusive_group(required=True)
    stop_group.add_argument('-p', '--precision', type=check_positive, required=False,
                            help='The precision to reach.')
    stop_group.add_argument('-i', '--iterations', type=check_positive_int, required=False,
                            help='The maximum number of iterations.')

    return parser
