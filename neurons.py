import re

import numpy as np

class LIFLayer:
    def __init__(self, num_neurons, tau_m=20.0, v_rest=-70.0, v_thresh=-50.0, v_reset=-70.0, t_ref=2.0, noise_std=1.0, lateral_inhibition=0.0, num_dendrites=10):
        if (
            not isinstance(num_dendrites, (int, np.integer))
            or isinstance(num_dendrites, bool)
            or num_dendrites <= 0
        ):
            raise ValueError("num_dendrites debe ser un entero mayor que cero.")
        self.num_neurons = num_neurons
        self.num_dendrites = int(num_dendrites)
        self.tau_m = tau_m  
        self.v_rest = v_rest
        self.base_v_thresh = v_thresh
        self.v_reset = v_reset
        self.t_ref = t_ref
        self.noise_std = noise_std
        self.lateral_inhibition = lateral_inhibition

        self.v = np.full(num_neurons, v_rest)
        self.refractory_time = np.zeros(num_neurons)  

    def step(self, I_ext, neuromod=None, dt=1.0):
        if neuromod is not None:
            effective_v_thresh = neuromod.apply_attention(self.base_v_thresh)
            I_gaba = neuromod.apply_rest()
            current_noise_std = max(0.0, neuromod.apply_affection(self.noise_std))
        else:
            effective_v_thresh = self.base_v_thresh
            I_gaba = 0.0
            current_noise_std = self.noise_std

        if (
            current_noise_std == 0.0
            and I_gaba == 0.0
        ):
            active_neurons = np.flatnonzero(
                (I_ext != 0.0)
                | (self.v != self.v_rest)
                | (self.refractory_time > 0.0)
            )
            spikes = np.zeros(self.num_neurons, dtype=int)
            if active_neurons.size == 0:
                return spikes

            available = active_neurons[
                self.refractory_time[active_neurons] <= 0.0
            ]
            dv = (
                -(self.v[available] - self.v_rest) + I_ext[available]
            ) / self.tau_m
            self.v[available] += dv * dt
            fired = active_neurons[self.v[active_neurons] >= effective_v_thresh]
            spikes[fired] = 1
            if self.lateral_inhibition > 0 and fired.size:
                self.v[spikes == 0] -= self.lateral_inhibition * fired.size
            self.v[fired] = self.v_reset
            self.refractory_time[fired] = self.t_ref
            self.refractory_time[active_neurons] = np.maximum(
                0.0,
                self.refractory_time[active_neurons] - dt,
            )
            return spikes

        not_refractory = self.refractory_time <= 0
        noise = (
            np.random.normal(0, current_noise_std, size=self.num_neurons)
            if current_noise_std > 0.0
            else 0.0
        )
        I_total = I_ext + I_gaba + noise

        dv = (-(self.v[not_refractory] - self.v_rest) + I_total[not_refractory]) / self.tau_m
        self.v[not_refractory] += dv * dt

        spikes = self.v >= effective_v_thresh

        if self.lateral_inhibition > 0 and np.any(spikes):
            self.v[~spikes] -= self.lateral_inhibition * np.sum(spikes)

        self.v[spikes] = self.v_reset
        self.refractory_time[spikes] = self.t_ref
        self.refractory_time = np.maximum(0, self.refractory_time - dt)

        return spikes.astype(int)


class TextSensoryLayer(LIFLayer):
    """Codifica presencia de tokens en una población de entrada LIF."""

    def __init__(
        self,
        num_neurons,
        vocabulary,
        tau_m=20.0,
        v_rest=-70.0,
        v_thresh=-50.0,
        t_ref=2.0,
        noise_std=0.2,
        lateral_inhibition=0.0,
        num_dendrites=10,
    ):
        super().__init__(
            num_neurons,
            tau_m=tau_m,
            v_rest=v_rest,
            v_thresh=v_thresh,
            t_ref=t_ref,
            noise_std=noise_std,
            lateral_inhibition=lateral_inhibition,
            num_dendrites=num_dendrites,
        )
        self.vocabulary = tuple(vocabulary)
        if len(self.vocabulary) != num_neurons or len(set(self.vocabulary)) != num_neurons:
            raise ValueError("El vocabulario debe tener un token único por neurona.")
        self.token_to_index = {
            token: index for index, token in enumerate(self.vocabulary)
        }

    @staticmethod
    def tokenize(text):
        return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

    def encode(self, text, max_current=80.0):
        if not isinstance(text, str):
            raise ValueError("La entrada de texto debe ser una cadena.")
        tokens = self.tokenize(text)
        if not tokens:
            raise ValueError("El texto debe contener al menos un token.")
        values = np.zeros(self.num_neurons, dtype=float)
        for token in tokens:
            index = self.token_to_index.get(
                token, self.token_to_index.get("<UNK>")
            )
            if index is not None:
                values[index] = 1.0
        return values * max_current


class ThalamicLayer(LIFLayer):
    """
    Tálamo Humano: Filtro del Núcleo Reticular (TRN) + Disparo en Ráfaga (Canales Ca2+ tipo T).
    Ahorra CPU bloqueando señales estáticas antes de que lleguen a la corteza.
    """
    def __init__(self, num_neurons, tau_m=10.0, v_rest=-65.0, v_thresh=-50.0, t_ref=1.0, noise_std=0.2, lateral_inhibition=5.0, gate_threshold=0.10, num_dendrites=10):
        super().__init__(num_neurons, tau_m, v_rest, v_thresh, v_reset=v_rest, t_ref=t_ref, noise_std=noise_std, lateral_inhibition=lateral_inhibition, num_dendrites=num_dendrites)
        self.gate_threshold = gate_threshold
        self.sensory_adaptation = np.zeros(num_neurons, dtype=float)
        self.burst_ready = np.ones(num_neurons, dtype=bool)  # Canales T listos tras silencio

    def step(self, I_ext, neuromod=None, dt=1.0):
        # AHORRO DE CPU: Si no hay entrada externa, salimos de inmediato sin calcular física
        if np.max(np.abs(I_ext)) < 1e-3 and np.all(self.v <= self.v_rest + 0.5):
            self.sensory_adaptation *= 0.9
            self.burst_ready[:] = True
            return np.zeros(self.num_neurons, dtype=int)

        ach = neuromod.attention_level if neuromod is not None else 0.5
        
        # Compuerta TRN modulada por Acetilcolina (Atención)
        trn_gate = np.clip(0.35 + 0.85 * ach, 0.2, 1.2)
        
        # Impulso de canales de Calcio tipo T (Modo Burst al despertar de la inhibición)
        t_channel_boost = np.where(self.burst_ready & (I_ext > 10.0), 12.0, 0.0)
        I_gated = (I_ext * trn_gate) + t_channel_boost - self.sensory_adaptation

        # AHORRO DE CPU: Si la señal fue filtrada por completo por falta de novedad
        if ach < self.gate_threshold and np.max(I_gated) < 4.0:
            self.v[:] = self.v_rest
            self.burst_ready[:] = True
            self.sensory_adaptation = 0.92 * self.sensory_adaptation + 0.08 * (I_ext * 0.3)
            return np.zeros(self.num_neurons, dtype=int)

        spikes = super().step(I_gated, neuromod=neuromod, dt=dt)

        # Actualizar estado de canales T y adaptación sensorial
        self.burst_ready = (spikes == 0) & (self.v <= self.v_rest + 2.0)
        if ach < 0.25:
            self.sensory_adaptation += spikes * 4.0
        self.sensory_adaptation *= np.exp(-dt / 30.0)

        return spikes

class DentateGranuleLayer(LIFLayer):
    """
    Giro Dentado (Hipocampo): Separación de Patrones mediante k-Winners-Take-All (k-WTA).
    Garantiza ultra-esparsidad (<15% de neuronas activas) para no saturar la memoria.
    """
    def __init__(self, num_neurons, tau_m=15.0, v_rest=-75.0, v_thresh=-45.0, t_ref=3.0, noise_std=0.1, lateral_inhibition=30.0, max_active_ratio=0.25, num_dendrites=10):
        super().__init__(num_neurons, tau_m, v_rest, v_thresh, v_reset=v_rest, t_ref=t_ref, noise_std=noise_std, lateral_inhibition=lateral_inhibition, num_dendrites=num_dendrites)
        # Número máximo de neuronas que pueden disparar al mismo tiempo (mínimo 1)
        self.k_winners = max(1, int(np.ceil(num_neurons * max_active_ratio)))

    def step(self, I_ext, neuromod=None, dt=1.0):
        # AHORRO DE CPU: Silencio instantáneo si la entrada es nula
        if np.max(np.abs(I_ext)) < 1e-3 and np.all(self.v <= self.v_rest + 0.5):
            return np.zeros(self.num_neurons, dtype=int)

        if neuromod is not None:
            effective_v_thresh = neuromod.apply_attention(self.base_v_thresh)
            I_gaba = neuromod.apply_rest()
        else:
            effective_v_thresh = self.base_v_thresh
            I_gaba = 0.0

        not_refractory = self.refractory_time <= 0
        dv = (-(self.v[not_refractory] - self.v_rest) + (I_ext[not_refractory] + I_gaba)) / self.tau_m
        self.v[not_refractory] += dv * dt

        # Candidatas a disparar
        candidate_spikes = self.v >= effective_v_thresh
        spikes = np.zeros(self.num_neurons, dtype=int)

        if np.any(candidate_spikes):
            # Competición estricta k-Winners-Take-All: solo ganan las de mayor voltaje
            num_candidates = np.sum(candidate_spikes)
            if num_candidates <= self.k_winners:
                spikes[candidate_spikes] = 1
            else:
                top_k_indices = np.argsort(self.v)[-self.k_winners:]
                spikes[top_k_indices] = 1

            # Inhibición lateral masiva sobre las perdedoras (Interneuronas en cesto)
            losers = spikes == 0
            self.v[losers] = np.maximum(self.v_rest, self.v[losers] - self.lateral_inhibition)
            self.v[spikes == 1] = self.v_reset
            self.refractory_time[spikes == 1] = self.t_ref

        self.refractory_time = np.maximum(0, self.refractory_time - dt)
        return spikes

class PyramidalLayer(LIFLayer):
    """
    Corteza Piramidal Humana: Adaptación de Frecuencia de Disparo (SFA - Corriente I_w)
    y amplificación dendrítica apical. Evita que una sola neurona monopolice el gasto energético.
    """
    def __init__(self, num_neurons, tau_m=20.0, v_rest=-70.0, v_thresh=-50.0, t_ref=2.0, noise_std=0.4, lateral_inhibition=15.0, sfa_strength=6.0, tau_w=80.0, num_dendrites=10):
        super().__init__(num_neurons, tau_m, v_rest, v_thresh, v_reset=v_rest, t_ref=t_ref, noise_std=noise_std, lateral_inhibition=lateral_inhibition, num_dendrites=num_dendrites)
        self.w_adapt = np.zeros(num_neurons, dtype=float)  # Corriente de fatiga individual (K+ lento)
        self.sfa_strength = sfa_strength
        self.tau_w = tau_w

    def step(self, I_ext, neuromod=None, dt=1.0):
        # Decaimiento natural de la adaptación de frecuencia
        self.w_adapt *= np.exp(-dt / self.tau_w)

        # AHORRO DE CPU: Si la corteza está en reposo y sin corriente entrante, saltar
        if np.max(np.abs(I_ext)) < 1e-3 and np.all(self.v <= self.v_rest + 0.5):
            return np.zeros(self.num_neurons, dtype=int)

        # La Acetilcolina (ACh) reduce la fatiga de adaptación en el cerebro humano para mantener el foco
        ach = neuromod.attention_level if neuromod is not None else 0.0
        effective_adaptation = self.w_adapt * (1.0 - 0.5 * ach)

        # Restamos la fatiga individual de cada neurona piramidal a su corriente de entrada
        I_cortical = I_ext - effective_adaptation
        spikes = super().step(I_cortical, neuromod=neuromod, dt=dt)

        # Las neuronas que dispararon acumulan fatiga de potasio (SFA)
        if np.any(spikes):
            self.w_adapt += spikes * self.sfa_strength

        return spikes

class StriatalMSNLayer(LIFLayer):
    """
    Estriado (Neuronas Espinosas Medianas - MSN): Bi-estabilidad (Down-state / Up-state)
    y modulación por receptores de Dopamina D1 (Vía Directa GO) + Winner-Take-All motor.
    """
    def __init__(self, num_neurons, tau_m=25.0, v_rest=-80.0, v_thresh=-55.0, t_ref=2.0, noise_std=0.1, lateral_inhibition=28.0, num_dendrites=10):
        super().__init__(num_neurons, tau_m, v_rest, v_thresh, v_reset=v_rest, t_ref=t_ref, noise_std=noise_std, lateral_inhibition=lateral_inhibition, num_dendrites=num_dendrites)

    def step(self, I_ext, neuromod=None, dt=1.0):
        # 1. FILTRO DE ESTADO DOWN (-80 mV): Las MSN tienen canales K+ (Kir) que bloquean
        # corrientes débiles o ruido. Si no hay empuje cortical fuerte, ahorramos CPU.
        if np.max(I_ext) < 8.0 and np.all(self.v <= self.v_rest + 1.0):
            return np.zeros(self.num_neurons, dtype=int)

        # 2. MODULACIÓN DOPAMINÉRGICA D1 (Transición al Up-state):
        # La dopamina facilita que las neuronas motoras salgan del Down-state (-80mV) y decidan.
        da = neuromod.DA if neuromod is not None else 0.2
        d1_boost = 1.0 + 0.35 * da
        
        # Resistencia de canales Kir en el Down-state (frena entradas tibias, deja pasar las fuertes)
        kir_leak = np.where(self.v < -68.0, 4.0, 0.0)
        I_striatal = (I_ext * d1_boost) - kir_leak

        spikes = super().step(I_striatal, neuromod=neuromod, dt=dt)

        # 3. SELECCIÓN DE ACCIÓN LIMPIA: Si dos acciones empatan en el mismo milisegundo,
        # el núcleo estriatal prioriza la que recibió mayor empuje neto
        if np.sum(spikes) > 1:
            winner = np.argmax(I_striatal)
            spikes[:] = 0
            spikes[winner] = 1
            self.v[:] = self.v_rest
            self.refractory_time[winner] = self.t_ref

        return spikes

class RetinalVisualLayer(LIFLayer):
    def __init__(self, width, height, tau_m=10.0, v_rest=-65.0, v_thresh=-50.0, t_ref=1.5, noise_std=0.2, lateral_inhibition=8.0, num_dendrites=10):
        self.width = width
        self.height = height
        super().__init__(width * height, tau_m, v_rest, v_thresh, v_reset=v_rest, t_ref=t_ref, noise_std=noise_std, lateral_inhibition=lateral_inhibition, num_dendrites=num_dendrites)
    def encode(self, image_2d, contrast_boost=True, max_current=80.0):
        img = np.clip(np.array(image_2d, dtype=float), 0.0, 1.0)
        if contrast_boost and img.ndim == 2:
            img = np.clip(img - 0.25 * np.mean(img), 0.0, 1.0)
        return img.flatten() * max_current

class CochlearAuditoryLayer(LIFLayer):
    def __init__(self, num_freq_bands, tau_m=6.0, v_rest=-65.0, v_thresh=-50.0, t_ref=1.0, noise_std=0.2, lateral_inhibition=5.0, num_dendrites=10):
        super().__init__(num_freq_bands, tau_m, v_rest, v_thresh, v_reset=v_rest, t_ref=t_ref, noise_std=noise_std, lateral_inhibition=lateral_inhibition, num_dendrites=num_dendrites)
    def encode(self, freq_spectrum, max_current=80.0):
        spectrum = np.clip(np.array(freq_spectrum, dtype=float), 0.0, 1.0) * max_current
        return spectrum

class SomatosensoryLayer(LIFLayer):
    def __init__(self, num_receptors, tau_m=12.0, v_rest=-68.0, v_thresh=-50.0, t_ref=2.0, noise_std=0.2, lateral_inhibition=6.0, num_dendrites=10):
        super().__init__(num_receptors, tau_m, v_rest, v_thresh, v_reset=v_rest, t_ref=t_ref, noise_std=noise_std, lateral_inhibition=lateral_inhibition, num_dendrites=num_dendrites)
        self.prev_pressure = np.zeros(num_receptors, dtype=float)
    def encode(self, pressure_map, max_current=80.0):
        p = np.clip(np.array(pressure_map, dtype=float).flatten(), 0.0, 1.0)
        delta_p = np.abs(p - self.prev_pressure)
        self.prev_pressure = p.copy()
        return np.clip(0.7 * p + 0.3 * delta_p, 0.0, 1.0) * max_current

class OlfactoryLayer(LIFLayer):
    def __init__(self, num_glomeruli, tau_m=25.0, v_rest=-70.0, v_thresh=-48.0, t_ref=3.0, noise_std=0.3, lateral_inhibition=20.0, num_dendrites=10):
        super().__init__(num_glomeruli, tau_m, v_rest, v_thresh, v_reset=v_rest, t_ref=t_ref, noise_std=noise_std, lateral_inhibition=lateral_inhibition, num_dendrites=num_dendrites)
    def encode(self, chemical_vector, max_current=80.0):
        c = np.clip(np.array(chemical_vector, dtype=float), 0.0, 1.0)
        return  np.log1p(c * (np.e - 1.0)) * max_current

class GustatoryLayer(LIFLayer):
    def __init__(self, num_taste_channels=5, tau_m=30.0, v_rest=-70.0, v_thresh=-50.0, t_ref=3.0, noise_std=0.2, lateral_inhibition=5.0, num_dendrites=10):
        super().__init__(num_taste_channels, tau_m, v_rest, v_thresh, v_reset=v_rest, t_ref=t_ref, noise_std=noise_std, lateral_inhibition=lateral_inhibition, num_dendrites=num_dendrites)
    def encode(self, taste_profile, max_current=80.0):
        return np.clip(np.array(taste_profile, dtype=float), 0.0, 1.0) * max_current
