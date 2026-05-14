import math

from simses.battery.cell import CellType
from simses.battery.derating import CurrentDerating
from simses.battery.state import BatteryState
from simses.degradation.degradation import DegradationModel


class Battery:
    """Battery system composed of cells in a series-parallel circuit.

    Models a pack of identical cells arranged as ``(serial, parallel)`` using
    an equivalent-circuit model (ECM): terminal voltage is
    ``OCV(SOC,T) + hysteresis + Rint × I``. Each call to :meth:`step` solves
    the equilibrium current for a requested power setpoint, clamps it to the
    hard limits (C-rate, voltage window, SOC window), optionally applies a
    :class:`~simses.battery.derating.CurrentDerating` strategy, and updates
    the state.

    Composition: a :class:`~simses.battery.cell.CellType` supplies the
    electrochemistry and per-cell physical properties; an optional
    :class:`~simses.degradation.degradation.DegradationModel` accumulates
    capacity fade and resistance rise; an optional ``CurrentDerating``
    reduces current in voltage or temperature derating zones.

    Sign convention: positive power / current = charging, negative =
    discharging.
    """

    def __init__(
        self,
        cell: CellType,
        circuit: tuple[int, int],  # (s, p)
        initial_states: dict,
        soc_limits: tuple[float, float] = (0.0, 1.0),  # in p.u.
        degradation: DegradationModel | bool | None = None,
        derating: CurrentDerating | None = None,
        effective_cooling_area: float = 1.0,
    ) -> None:
        """
        Args:
            cell: Cell model defining OCV, Rint, and physical parameters.
            circuit: Series-parallel configuration as ``(s, p)``.
            initial_states: Dict with keys ``start_soc``, ``start_T``, and
                optionally ``start_soh_Q`` / ``start_soh_R``.
            soc_limits: ``(soc_min, soc_max)`` operating window in p.u.
            degradation: Degradation model, or ``True`` to use the cell's
                default (fresh battery, no prior aging history), or ``None`` /
                ``False`` to disable. To warm-start from a known degradation
                state, pass an explicit ``DegradationModel`` constructed with
                an ``initial_state``.
            derating: Optional current-derating strategy applied after hard
                limits.
            effective_cooling_area: Fraction of the total cell surface area
                that participates in heat exchange with the environment, in
                p.u. (default 1.0 = full surface area).  Use values below 1
                to model packs where only a portion of each cell face is
                exposed to coolant, e.g. 0.5 for a two-sided cooling plate
                that covers half the cell surface.
        """
        if degradation is False:
            degradation = None
        if degradation is True:
            initial_soc = initial_states["start_soc"]
            degradation = cell.default_degradation_model(initial_soc)
            if degradation is None:
                raise ValueError(
                    f"{type(cell).__name__} has no default degradation model. "
                    "Pass an explicit DegradationModel or use degradation=None."
                )
        self.cell = cell
        self.circuit = circuit
        self.soc_limits = soc_limits
        self.degradation = degradation
        self.derating = derating
        self.effective_cooling_area = effective_cooling_area
        self.state = self.initialize_state(**initial_states)

    def initialize_state(
        self, start_soc: float, start_T: float, start_soh_Q: float = 1.0, start_soh_R: float = 1.0
    ) -> BatteryState:
        """Create the initial battery state from starting conditions.

        Sets SOC, temperature, and SoH from the arguments, then evaluates
        OCV, hysteresis, Rint, and entropic coefficient at that initial
        state so the returned object is consistent before the first
        :meth:`step` call.

        Args:
            start_soc: Initial state of charge in p.u.
            start_T: Initial cell temperature in °C.
            start_soh_Q: Initial capacity SoH in p.u. (default 1.0 = fresh).
            start_soh_R: Initial resistance SoH in p.u. (default 1.0 = fresh).

        Returns:
            A fully-initialised :class:`BatteryState`.
        """
        state = BatteryState(
            v=0,  # uninitialized
            i=0,  # uninitialized
            T=start_T,
            power=0,
            power_setpoint=0,
            loss=0,
            heat=0,
            soc=start_soc,
            ocv=0,  # uninitialized
            hys=0,  # uninitialized
            entropy=0,  # uninitialized
            is_charge=True,
            rint=0,  # uninitialized
            soh_Q=start_soh_Q,
            soh_R=start_soh_R,
            i_max_charge=0.0,
            i_max_discharge=0.0,
        )
        state.ocv = state.v = self.open_circuit_voltage(state)
        state.hys = self.hysteresis_voltage(state)
        state.rint = self.internal_resistance(state)
        state.entropy = self.entropic_coefficient(state)
        return state

    def step(self, power_setpoint: float, dt: float, faststep_factor:int = None) -> None:
        """Advance the battery state by one timestep.

        If the battery cannot fulfil the power setpoint due to hard limits
        (C-rate, voltage window, SOC window) or optional derating, the
        current is curtailed and ``state.power`` reflects what was actually
        delivered — not the original setpoint.

        Args:
            power_setpoint: Requested power in W. Positive = charging,
                negative = discharging.
            dt: Timestep in seconds.
            faststep_factor: Changes battery behaviour to split power calculation into the passed number of smaller steps,
                this will ignore derating
        """
        state: BatteryState = self.state
        state.is_charge = power_setpoint > 0.0

        # --- phase 1: refresh derived cell properties from current soc/T ---
        # ocv, hys, rint are derived from inputs (soc, T, soh_R) that do not
        # change during this method, so updating them here is safe and ensures
        # all calculations — including derating — use consistent current values.
        ocv = state.ocv = self.open_circuit_voltage(state)
        hys = state.hys = self.hysteresis_voltage(state)
        rint = state.rint = self.internal_resistance(state)
        entropy = state.entropy = self.entropic_coefficient(state)
        Q = self.capacity(state)
        soc = state.soc

        # 1. Calculate equilibrium current to meet power setpoint
        i = self.equilibrium_current(power_setpoint, ocv, hys, rint)

        # 2. Calculate hard current limits (C-rate, voltage, SOC)
        i_max_charge, i_max_discharge = self.calculate_max_currents(soc, dt, ocv, hys, rint, Q)

        # Do full single step if no substepping is requested
        if faststep_factor is None:
            # 3. Curtail solved current to hard limits
            if i > 0:
                i = min(i, i_max_charge)
            elif i < 0:
                i = max(i, i_max_discharge)

            # 4. Apply derating (optional).
            # i_max_charge / i_max_discharge are only updated when derating actually reduces i,
            # so that the reported limits reflect the hard limits during normal operation and only
            # drop when the battery is genuinely in the derating zone.
            if self.derating is not None:
                i_derate = self.derating.derate(i, state)
                if i > 0 and i_derate < i:
                    i = i_derate
                    i_max_charge = min(i_max_charge, i_derate)
                elif i < 0 and i_derate > i:
                    i = i_derate
                    i_max_discharge = max(i_max_discharge, i_derate)

            # update soc
            (soc_min, soc_max) = self.soc_limits
            soc += i * dt / Q / 3600
            soc = max(soc_min, min(soc, soc_max))

            # check current direction, maintain previous state if in rest
            is_charge = state.is_charge if i == 0 else i > 0

            # update terminal voltage and power
            v = ocv + hys + rint * i
            power = v * i

            # update losses
            loss_irr = (v - ocv) * i  # irreversible losses
            loss_rev = entropy * (state.T + 273.15) * i  # reversible losses (T must be absolute)
            heat = loss_irr + loss_rev  # internal heat generation


        # Alternative power calculations for n = faststep_factor smaller substeps
        else:
            sub_dt = dt / faststep_factor

            # Scaled down losses will be accumulated during substeps
            entropy_factor = entropy * (state.T + 273.15)
            loss_irr = 0
            loss_rev = 0

            # Look up soc limits
            (soc_min, soc_max) = self.soc_limits
            soc_factor = Q / (sub_dt / 3600)

            # Curtail solved current to static limits once
            if i > 0:
                i = min(
                    i,
                    self.max_charge_current,
                    (self.max_voltage - ocv - hys) / rint
                )
            elif i < 0:
                i = max(
                    i,
                    -self.max_discharge_current,
                    (self.min_voltage - ocv - hys) / rint
                )


            for _substep in range(faststep_factor):
                # 3. Curtail solved current to soc limit
                if i > 0:
                    i = min(i, (soc_max - soc) * soc_factor)
                elif i < 0:
                    i = max(i, (soc_min - soc) * soc_factor)

                # 4. Derating is omitted, it depends on the battery state which is not updated during substeps

                # Update soc
                soc += i * sub_dt / Q / 3600
                soc = max(soc_min, min(soc, soc_max))

                # Add losses relative to substep size
                loss_irr += (hys + rint * i) * i / faststep_factor
                loss_rev += entropy_factor * i / faststep_factor


            # check current direction, maintain previous state if in rest
            is_charge = state.is_charge if i == 0 else i > 0

            # update terminal voltage and power
            v = ocv + hys + rint * i
            power = v * i

            # update losses
            heat = loss_irr + loss_rev

        # --- phase 2: write output state ---
        state.v = v
        state.i = i
        state.power = power
        state.power_setpoint = power_setpoint
        state.loss = loss_irr
        state.heat = heat
        state.soc = soc
        state.is_charge = is_charge
        state.i_max_charge = i_max_charge
        state.i_max_discharge = i_max_discharge

        if self.degradation is not None:
            self.degradation.step(self.state, dt)  # updates state.soh_Q and state.soh_R

    def equilibrium_current(self, power_setpoint: float, ocv: float, hys: float, rint: float) -> float:
        """Solve the ECM for the current that meets a power setpoint.

        Solves the quadratic ``P = I × (OCV + hys + Rint × I)`` for ``I``
        and returns the physically meaningful (positive-discriminant) root.

        Args:
            power_setpoint: Target power in W.
            ocv: System open-circuit voltage in V.
            hys: System hysteresis voltage in V.
            rint: System internal resistance in Ω.

        Returns:
            Equilibrium current in A. Positive = charging, negative =
            discharging.
        """
        ocv = ocv + hys  # include hysteresis in equilibrium calculation
        if power_setpoint == 0.0:
            return 0.0
        return -(ocv - math.sqrt(ocv**2 + 4 * rint * power_setpoint)) / (2 * rint)

    def calculate_max_currents(
        self, soc: float, dt: float, ocv: float, hys: float, rint: float, Q: float
    ) -> tuple[float, float]:
        """Return the allowed current window for the next timestep.

        Each bound is the most restrictive of three limits: the C-rate
        limit (from cell ``max_charge_rate`` / ``max_discharge_rate``), the
        voltage limit (current that would drive terminal voltage to
        ``max_voltage`` or ``min_voltage`` this step), and the SOC limit
        (current that would drive SOC to the configured ``soc_limits``
        this step).

        Args:
            soc: Battery state of charge.
            dt: Timestep in seconds.
            ocv: System open-circuit voltage in V.
            hys: System hysteresis voltage in V.
            rint: System internal resistance in Ω.
            Q: Current capacity in Ah (scaled by ``soh_Q``).

        Returns:
            Tuple ``(i_max_charge, i_max_discharge)`` in A. Charge bound
            is non-negative; discharge bound is non-positive.
        """
        (soc_min, soc_max) = self.soc_limits

        # charge (all three values are positive; min = most restrictive)
        i_max_charge = min(
            self.max_charge_current,  # C-rate limit
            (self.max_voltage - ocv - hys) / rint,  # voltage limit
            (soc_max - soc) * Q / (dt / 3600),  # SOC limit
        )
        # discharge (all three values are negative; max = least negative = most restrictive)
        i_max_discharge = max(
            -self.max_discharge_current,  # C-rate limit
            (self.min_voltage - ocv - hys) / rint,  # voltage limit
            (soc_min - soc) * Q / (dt / 3600),  # SOC limit
        )
        return i_max_charge, i_max_discharge

    ## electrical properties
    def open_circuit_voltage(self, state: BatteryState) -> float:
        """Return the system-level open-circuit voltage in V."""
        (serial, parallel) = self.circuit

        return self.cell.open_circuit_voltage(state) * serial

    def hysteresis_voltage(self, state: BatteryState) -> float:
        """Return the system-level hysteresis voltage in V."""
        (serial, parallel) = self.circuit

        return self.cell.hysteresis_voltage(state) * serial

    def internal_resistance(self, state: BatteryState) -> float:
        """Return the system-level internal resistance in Ohms, scaled by SoH."""
        (serial, parallel) = self.circuit

        # state.i = state.i / parallel # <- should be scaled to the cell
        return self.cell.internal_resistance(state) / parallel * serial * state.soh_R

    def entropic_coefficient(self, state: BatteryState) -> float:
        """Return the system-level entropic coefficient in V/K."""
        (serial, parallel) = self.circuit
        return self.cell.entropic_coefficient(state) * serial

    def capacity(self, state: BatteryState) -> float:
        """Return the current capacity in Ah, scaled by SoH."""
        return self.nominal_capacity * state.soh_Q

    def energy_capacity(self, state: BatteryState) -> float:
        """Return the current energy capacity in Wh, scaled by SoH."""
        return self.nominal_energy_capacity * state.soh_Q

    @property
    def nominal_capacity(self) -> float:
        """Nominal capacity of the battery system in Ah."""
        (serial, parallel) = self.circuit

        return self.cell.electrical.nominal_capacity * parallel

    @property
    def nominal_voltage(self) -> float:
        """Nominal voltage of the battery system in V."""
        (serial, parallel) = self.circuit

        return self.cell.electrical.nominal_voltage * serial

    @property
    def nominal_energy_capacity(self) -> float:
        """Nominal energy capacity of the battery system in Wh."""
        return self.nominal_capacity * self.nominal_voltage

    @property
    def min_voltage(self) -> float:
        """Minimum allowed voltage of the battery system in V."""
        (serial, parallel) = self.circuit

        return self.cell.electrical.min_voltage * serial

    @property
    def max_voltage(self) -> float:
        """Maximum allowed voltage of the battery system in V."""
        (serial, parallel) = self.circuit

        return self.cell.electrical.max_voltage * serial

    @property
    def max_charge_current(self) -> float:
        """Maximum allowed charge current in A."""
        (serial, parallel) = self.circuit

        return self.cell.electrical.nominal_capacity * self.cell.electrical.max_charge_rate * parallel

    @property
    def max_discharge_current(self) -> float:
        """Maximum allowed discharge current in A."""
        (serial, parallel) = self.circuit

        return self.cell.electrical.nominal_capacity * self.cell.electrical.max_discharge_rate * parallel

    @property
    def coulomb_efficiency(self) -> float:
        """Coulomb efficiency of the cell in p.u."""
        return self.cell.electrical.coulomb_efficiency

    ## thermal properties
    @property
    def thermal_capacity(self) -> float:
        """Total thermal capacity of the battery system in J/K."""
        (serial, parallel) = self.circuit

        return self.cell.thermal.specific_heat * self.cell.thermal.mass * serial * parallel

    @property
    def convection_coefficient(self) -> float:
        """Convection coefficient of the cell in W/m2K."""
        return self.cell.thermal.convection_coefficient

    @property
    def thermal_resistance(self) -> float:
        """Thermal resistance of the battery system in K/W."""
        return 1 / (self.convection_coefficient * self.area)

    @property
    def min_temperature(self) -> float:
        """Minimum allowed temperature in °C."""
        return self.cell.thermal.min_temperature

    @property
    def max_temperature(self) -> float:
        """Maximum allowed temperature in °C."""
        return self.cell.thermal.max_temperature

    @property
    def area(self) -> float:
        """Effective cooling area of the pack in m².

        Equals the per-cell surface area (``cell.format.area``) scaled by the
        ``effective_cooling_area`` fraction and the pack size
        ``(serial × parallel)``. Used by :attr:`thermal_resistance` to compute
        the convective coupling between the pack and the thermal environment.
        """
        (serial, parallel) = self.circuit

        return self.cell.format.area * self.effective_cooling_area * serial * parallel
