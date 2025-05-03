from enum import Enum
from re import S
import re
from typing import Any
import keras

class SignalType(Enum):
    """Enum for signal types."""
    CorrientePlasma = 1,
    ModeLock = 2,
    Inductancia = 3,
    Densidad = 4,
    DerivadaEnergiaDiamagnetica = 5,
    PotenciaRadiada = 6,
    PotenciaDeEntrada = 7,

def get_signal_type(signal_type: int) -> SignalType:
    """
    Get the signal type from an integer.
    :param signal_type: The signal type as an integer.
    :return: The signal type as a SignalType enum.
    """
    return SignalType(signal_type) if signal_type in [s.value for s in SignalType] else None

class DisruptionClass(Enum):
    """Enum for disruption classes."""
    Normal = 0,
    Anomaly = 1,
    Unknown = 2,


class Signal:
    """Class representing a signal."""

    def __init__(self, label: str, times: list[float], values: list[float], signal_type: SignalType, disruption_class=DisruptionClass.Unknown):
        """
        Initialize a Signal object.
        :param label: The label of the signal. It is the file name
        :param times: The time values of the signal.
        :param values: The values of the signal.
        :param signal_type: The type of the signal.
        :param disruption_class: The class of the disruption.
        """
        self.label = label
        self.times = times
        self.values = values
        self.disruption_class = disruption_class
        self.signal_type = signal_type
        self.min = min(values) 
        self.max = max(values)

    def normalize(self, min_of_its_type: Any | None, max_of_its_type: Any | None):
        """
        Normalize the signal values to the range [0, 1]. Admits min and max values to normalize against them.
        :param min_of_its_type: The minimum value of the signal type.
        :param max_of_its_type: The maximum value of the signal type.
        """
        min = min_of_its_type if min_of_its_type is not None else self.min
        max = max_of_its_type if max_of_its_type is not None else self.max
    
        self.values = [(value - min) / (max - min) for value in self.values]


class Discharge:
    """Class representing a discharge."""

    def __init__(self, signals: list[Signal], disruption_class=DisruptionClass.Unknown):
        self.signals = signals
        self.disruption_class = disruption_class
        self.is_padded = False
        self.is_normalized = False

def normalize_vec(list_values: list[Signal]):
    """
    Normalize a list of signals.
    :param list_values: The list of signals to normalize.
    """
    signals_normalized = []
    min_by_type = {}
    max_by_type = {}

    for signal in list_values:
        if signal.signal_type not in min_by_type:
            min_by_type[signal.signal_type] = signal.min
            max_by_type[signal.signal_type] = signal.max
        else:
            min_by_type[signal.signal_type] = min(min_by_type[signal.signal_type], signal.min)
            max_by_type[signal.signal_type] = max(max_by_type[signal.signal_type], signal.max)


    for signal in list_values:
        signal.normalize(min_by_type[signal.signal_type], max_by_type[signal.signal_type])
        signals_normalized.append(signal)

    return signals_normalized

def normalize(discharges: list[Discharge]) -> list[Discharge]:
    """
    Normalize the signals in a list of discharges.
    :param discharges: The list of discharges to normalize.
    :return: The normalized discharges.
    """
    all_signals = []
    for discharge in discharges:
        all_signals += discharge.signals

    normalized_signals = normalize_vec(all_signals)

    for discharge in discharges:
        for i, signal in enumerate(discharge.signals):
            discharge.signals[i].values = normalized_signals[i].values
        discharge.is_normalized = True

    return discharges

def are_normalized(discharges: list[Discharge]) -> bool:
    return all([discharge.is_normalized for discharge in discharges])

def pad(discharges: list[Discharge]) -> list[Discharge]:
    """
    Pad the signals with zeros in a list of discharges to the same length.
    :param discharges: The list of discharges to pad.
    :return: The padded discharges.
    """
    max_length = max([len(signal.values) for discharge in discharges for signal in discharge.signals])
    for discharge in discharges:
        for signal in discharge.signals:
            if len(signal.values) < max_length:
                signal.values += [0] * (max_length - len(signal.values))
        discharge.is_padded = True
    return discharges

def are_padded(discharges: list[Discharge]) -> bool:
    """
    Check if the signals in a list of discharges are padded.
    :param discharges: The list of discharges to check.
    :return: True if the signals are padded, False otherwise.
    """
    return all([discharge.is_padded for discharge in discharges])

def get_X_y(discharges: list[Discharge]) -> tuple[list[list[float]], list[int]]:
    """
    Get the X and y values from a list of discharges.
    They are parallel lists, where X is the list of signals and y is the list of disruption classes.
    :param discharges: The list of discharges to get the X and y values from.
    :return: The X and y values.
    """
    if not are_normalized(discharges):
        discharges = normalize(discharges)

    # if not are_padded(discharges):
    #     discharges = pad(discharges)
    
    # X es una lista de listas
    # - La lista interna contiene los valores de la señal
    # - La lista externa contiene todas las señales
    # y es paralelo a X: Contiene si es anomalia o no

    X = [[signal.values for signal in discharge.signals] for discharge in discharges]
    y = [discharge.disruption_class.value for discharge in discharges]

    return X, y