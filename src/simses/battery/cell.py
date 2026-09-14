from abc import ABC, abstractmethod

from simses.battery.format import CellFormat
from simses.battery.properties import ElectricalCellProperties, ThermalCellProperties
from simses.battery.state import BatteryState
from simses.degradation.degradation import DegradationModel
from simses.degradation.state import DegradationState


class CellType(ABC):
    """
    Abstract base class defining the interface and common properties for different types of battery cells.
    Subclasses must implement methods for calculating open-circuit voltage and internal resistance.
    Optionally, subclasses can override default_degradation_model() to ship a built-in degradation model.

    Attributes:
        electrical (ElectricalCellProperties): Electrical properties of the cell.
        thermal (ThermalCellProperties): Thermal properties of the cell.
        format (CellFormat): Physical format of the cell.
    Methods:
        open_circuit_voltage(state: BatteryState) -> float:
            Abstract method to compute the open-circuit voltage for a given battery state.
        hysteresis_voltage(state: BatteryState) -> float:
            Returns the hysteresis voltage for a given battery state. Default is 0.
        internal_resistance(state: BatteryState) -> float:
            Abstract method to compute the internal resistance (beginning of life) for a given battery state.
        self_discharge_current(state: BatteryState) -> float:
            Returns the self discharge current for a given battery state.
        default_degradation_model(initial_soc: float) -> DegradationModel | None:
            Returns the cell's built-in default degradation model, or None if not defined.
    """

    def __init__(
        self,
        electrical: ElectricalCellProperties,
        thermal: ThermalCellProperties,
        cell_format: CellFormat,
    ) -> None:
        super().__init__()
        self.electrical = electrical
        self.thermal = thermal
        self.format = cell_format

    @abstractmethod
    def open_circuit_voltage(self, state: BatteryState) -> float:
        """Compute the open-circuit voltage (in V) for a given battery state."""
        pass

    def hysteresis_voltage(self, state: BatteryState) -> float:
        """Compute the hysteresis voltage (in V) for a given battery state. Default is 0."""
        return 0.0

    @abstractmethod
    def internal_resistance(self, state: BatteryState) -> float:
        """Compute the beginning-of-life internal resistance (in Ohms) for a given battery state."""
        pass

    def self_discharge_current(self, state: BatteryState) -> float:
        """
        Compute the self-discharge-current (in A) for a given battery state.
        Default is None to indicate no self-discharge model is available.
        """
        return None

    def entropic_coefficient(self, state: BatteryState) -> float:
        """Compute entropic coefficient (in V/K) for a given battery state. Default is 0."""
        return 0.0

    @classmethod
    def default_degradation_model(
        cls,
        initial_soc: float,
        initial_state: DegradationState | None = None,
    ) -> DegradationModel | None:
        """Return the cell's built-in default degradation model, or None if not defined."""
        return None
