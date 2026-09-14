from dataclasses import dataclass
from typing import Any, Protocol


class ConverterLossModel(Protocol):
    """Protocol for AC/DC converter loss models.

    All power values are normalised to p.u. of max_power.
    Sign convention: positive = charging, negative = discharging.
    """

    def ac_to_dc(self, power_norm: float) -> float:
        """Convert normalised AC power to normalised DC power."""
        ...

    def dc_to_ac(self, power_norm: float) -> float:
        """Convert normalised DC power to normalised AC power."""
        ...


@dataclass
class ConverterState:
    """Mutable state of a :class:`Converter`.

    Attributes:
        power_setpoint: Requested AC power for the current timestep in W.
        power: AC power actually delivered this timestep in W
            (may differ from ``power_setpoint`` after storage-limited
            recomputation).
        loss: Conversion loss this timestep in W (``power - power_dc``).
    """

    power_setpoint: float = 0.0
    power: float = 0.0
    loss: float = 0.0


class Converter:
    """AC/DC converter that wraps a downstream storage with a loss model.

    Clamps the AC power setpoint to the rated ``max_power``, converts it to
    DC via the loss model, and forwards it to the storage. If the storage
    cannot fulfil the requested DC power (by more than 1%), the converter
    recomputes the actual AC power from the delivered DC power.

    The storage is duck-typed: any object with a ``step(power_setpoint, dt)``
    method and a ``state.power`` attribute — typically a :class:`Battery`,
    but also another :class:`Converter` (enabling converter chaining).

    Attributes:
        max_power: Rated maximum power of the converter in W.
        state: Current converter state (power, setpoint, loss).
        model: Loss model for AC/DC conversion.
        storage: Downstream storage receiving the DC power setpoint.
    """

    def __init__(self, loss_model: ConverterLossModel, max_power: float, storage: Any, tolerance: float = 0.01) -> None:
        """
        Args:
            loss_model: AC/DC loss model satisfying :class:`ConverterLossModel`.
            max_power: Rated maximum AC power of the converter in W (the
                normalisation base for ``loss_model``).
            storage: Downstream storage exposing ``step(power, dt)`` and
                ``state.power``. Typically a :class:`Battery`.
            tolerance: Relative difference between requested and
            actual DC power above which AC power is re-calculated
        """
        self.max_power = max_power
        self.state = ConverterState()
        self.model = loss_model
        self.storage = storage
        self.tolerance = abs(tolerance)

    def step(self, power_setpoint: float, dt: float) -> None:
        """Apply an AC power setpoint over one timestep.

        Two-pass resolution:
          1. Clamp the AC setpoint to ``[-max_power, max_power]``.
          2. Convert to DC via the loss model and delegate to
             ``storage.step(power_dc, dt)``.
          3. If the storage cannot fulfil the DC request (mismatch > 1%),
             recompute the actual AC power from the delivered DC power.

        Side effects: updates ``self.state`` with the resulting
        ``power_setpoint``, ``power`` (AC side), and ``loss``.

        Args:
            power_setpoint: Requested AC power in W. Positive = charging,
                negative = discharging.
            dt: Timestep in seconds.
        """
        max_power = self.max_power
        power_ac = max(-max_power, min(power_setpoint, max_power))
        power_dc = self.ac_to_dc(power_ac)

        self.storage.step(power_dc, dt)
        power_storage = self.storage.state.power

        # check if subsystem fulfilled DC power
        # if not, re-calculate required AC power
        if power_dc != 0 and (abs(power_dc - power_storage) / abs(power_dc)) > self.tolerance:
            power_dc = power_storage
            power_ac = self.dc_to_ac(power_dc)

        # calculate conversion losses
        loss = power_ac - power_dc

        # update state
        self.state.power_setpoint = power_setpoint
        self.state.power = power_ac
        self.state.loss = loss

    def ac_to_dc(self, power_ac: float) -> float:
        """Convert an AC power in W to the corresponding DC power in W.

        Forward direction used in the normal simulation flow: the AC setpoint
        is converted to the DC power delivered to the storage.

        Args:
            power_ac: AC power in W. Positive = charging, negative =
                discharging.

        Returns:
            DC power in W, same sign convention as ``power_ac``.
        """
        max_power = self.max_power
        return self.model.ac_to_dc(power_ac / max_power) * max_power

    def dc_to_ac(self, power_dc: float) -> float:
        """Convert a DC power in W to the corresponding AC power in W.

        Inverse of :meth:`ac_to_dc`. Used when the storage cannot fulfil
        the requested DC power — the delivered DC power is back-converted
        to find the AC power actually exchanged at the converter's AC
        terminals.

        Args:
            power_dc: DC power in W. Positive = charging, negative =
                discharging.

        Returns:
            AC power in W, same sign convention as ``power_dc``.
        """
        max_power = self.max_power
        return self.model.dc_to_ac(power_dc / max_power) * max_power
