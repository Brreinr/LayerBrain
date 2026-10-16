import numpy as np


class NeuromodulatorySystem:
    """Low-pass neuromodulator concentrations with signed reward prediction error."""

    DA_BASELINE = 0.15
    GABA_OVERLOAD_THRESHOLD = 0.25

    def __init__(
        self,
        dt=1.0,
        tau_ACh=100.0,
        tau_DA=200.0,
        tau_GABA=300.0,
        tau_OXT=400.0,
        gaba_baseline=0.2,
        route_attention_gain=0.8,
    ):
        self._validate_dt(dt)
        for name, tau in (
            ("tau_ACh", tau_ACh),
            ("tau_DA", tau_DA),
            ("tau_GABA", tau_GABA),
            ("tau_OXT", tau_OXT),
        ):
            if not np.isfinite(tau) or tau <= 0.0:
                raise ValueError(f"{name} debe ser finita y mayor que cero.")
        if not np.isfinite(gaba_baseline) or not 0.0 <= gaba_baseline <= 1.0:
            raise ValueError("gaba_baseline debe estar entre cero y uno.")
        if (
            not np.isfinite(route_attention_gain)
            or not 0.0 <= route_attention_gain <= 1.0
        ):
            raise ValueError("route_attention_gain debe estar entre cero y uno.")

        self.dt = dt
        self.tau_ACh = tau_ACh
        self.tau_DA = tau_DA
        self.tau_GABA = tau_GABA
        self.tau_OXT = tau_OXT
        self.gaba_baseline = gaba_baseline
        self.route_attention_gain = float(route_attention_gain)
        self.reset()

    @staticmethod
    def _validate_dt(dt):
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt debe ser finito y no negativo.")

    @staticmethod
    def _input(value, name, lower=0.0, upper=1.0):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} debe ser finito.")
        return float(np.clip(value, lower, upper))

    @staticmethod
    def _approach(current, target, tau, dt):
        return target + (current - target) * np.exp(-dt / tau)

    @property
    def reward_signal(self):
        if self.DA >= self.DA_BASELINE:
            return (self.DA - self.DA_BASELINE) / (1.0 - self.DA_BASELINE)
        return (self.DA - self.DA_BASELINE) / self.DA_BASELINE

    def reset(self):
        self.ACh = 0.0
        self.DA = self.DA_BASELINE
        self.GABA = self.gaba_baseline
        self.OXT = 0.0
        self.route_attention = 0.0

    @property
    def attention_level(self):
        """Combina atención sensorial y foco ganado por la ruta recompensada."""
        route_focus = self.route_attention_gain * self.route_attention
        return 1.0 - (1.0 - self.ACh) * (1.0 - route_focus)

    def set_route_attention(self, value):
        self.route_attention = self._input(value, "route_attention")

    def update_attention(self, novelty_input: float):
        target = self._input(novelty_input, "novelty_input")
        self.ACh = self._approach(self.ACh, target, self.tau_ACh, self.dt)

    def apply_attention(self, base_v_thresh: float, beta: float = 10.0) -> float:
        return base_v_thresh - beta * self.attention_level

    def update_motivation(self, reward_input: float):
        reward = self._input(reward_input, "reward_input", lower=-1.0, upper=1.0)
        target = np.clip(self.DA_BASELINE + reward, 0.0, 1.0)
        self.DA = self._approach(self.DA, target, self.tau_DA, self.dt)

    def apply_motivation(self, base_learning_rate: float) -> float:
        return base_learning_rate * max(0.0, self.reward_signal)

    def apply_reward_modulation(self, base_learning_rate: float) -> float:
        return base_learning_rate * self.reward_signal

    def update_rest(
        self,
        fatigue_input: float,
        overload_input: float = 0.0,
        uncertainty_input: float = 0.0,
        basal=None,
    ):
        fatigue = self._input(fatigue_input, "fatigue_input")
        overload = self._input(overload_input, "overload_input")
        uncertainty = self._input(uncertainty_input, "uncertainty_input")
        resting_level = self.gaba_baseline if basal is None else self._input(basal, "basal")
        combined_stress = 1.0 - (
            (1.0 - fatigue) * (1.0 - overload) * (1.0 - uncertainty)
        )
        target = resting_level + (1.0 - resting_level) * combined_stress
        tau = self.tau_GABA
        if target > self.GABA:
            tau = max(self.dt, self.tau_GABA * 0.1)
        self.GABA = self._approach(self.GABA, target, tau, self.dt)

    def apply_rest(self, g_max: float = 15.0) -> float:
        return -g_max * self.GABA

    def update_affection(self, social_input: float):
        target = self._input(social_input, "social_input")
        self.OXT = self._approach(self.OXT, target, self.tau_OXT, self.dt)

    def apply_affection(self, noise_std: float, gamma: float = 0.8) -> float:
        if not np.isfinite(noise_std) or noise_std < 0.0:
            raise ValueError("noise_std debe ser finito y no negativo.")
        gamma = self._input(gamma, "gamma")
        return noise_std * (1.0 - gamma * self.OXT)

    def step(
        self,
        novelty: float,
        reward: float,
        fatigue: float = 0.0,
        overload: float = 0.0,
        uncertainty: float = 0.0,
        social: float = 0.0,
        dt: float = 1.0,
    ):
        self._validate_dt(dt)
        self.dt = dt
        self.update_attention(novelty)
        self.update_motivation(reward)
        self.update_rest(fatigue, overload, uncertainty)
        self.update_affection(social)


# Backwards-compatible spelling used by earlier releases.
NeuromdulatorySystem = NeuromodulatorySystem
