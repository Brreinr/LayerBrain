"""Reproducible sparse-synapse scaling benchmark.

Run from the project root, for example:
    python benchmarks/benchmark_scaling.py --neurons 10000 --density 0.01
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layerbrain import LIFLayer, SparseReservoirSynapse


def process_memory_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        process = kernel32.GetCurrentProcess()
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_memory_info = psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_memory_info.restype = wintypes.BOOL
        success = get_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not success:
            error = ctypes.get_last_error()
            raise OSError(
                error,
                "GetProcessMemoryInfo no pudo leer la memoria del proceso.",
            )
        return int(counters.WorkingSetSize)

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def run_benchmark(neurons, density, steps, active_ratio, quantized):
    np.random.seed(2026)
    baseline_memory = process_memory_bytes()
    synapse = SparseReservoirSynapse(
        neurons,
        neurons,
        density=density,
        block_size=4096,
    )
    layer = LIFLayer(
        neurons,
        v_rest=0.0,
        v_thresh=1.0,
        v_reset=0.0,
        noise_std=0.0,
        t_ref=1.0,
    )
    float_connection_count = synapse.connection_count
    float_weight_bytes = synapse.weights.nbytes
    rng = np.random.default_rng(2026)
    active_count = max(1, int(neurons * active_ratio))
    elapsed = 0.0
    layer_elapsed = 0.0
    total_skipped = 0
    float_reference = []

    for step in range(steps):
        pre_spikes = np.zeros(neurons, dtype=np.float32)
        active = rng.choice(neurons, active_count, replace=False)
        pre_spikes[active] = 1.0
        started = time.perf_counter()
        current = synapse.step(pre_spikes, learn=False)
        elapsed += time.perf_counter() - started
        layer_input = np.zeros(neurons, dtype=np.float32)
        layer_input[active] = 2.0
        started = time.perf_counter()
        layer.step(layer_input)
        layer_elapsed += time.perf_counter() - started
        total_skipped += neurons - active.size
        if step < 20:
            float_reference.append((active.astype(np.int32), int(np.argmax(current))))
        synapse.g.fill(0.0)
        synapse._g_scale = 1.0

    agreement = None
    if quantized:
        synapse.quantize_weights()
        matches = 0
        for active, expected_winner in float_reference:
            synapse.g.fill(0.0)
            synapse._g_scale = 1.0
            pre_spikes = np.zeros(neurons, dtype=np.float32)
            pre_spikes[active] = 1.0
            actual_current = synapse.step(pre_spikes, learn=False)
            matches += int(np.argmax(actual_current) == expected_winner)
        agreement = matches / len(float_reference) if float_reference else 1.0

    rss = process_memory_bytes()
    return {
        "neurons_per_layer": neurons,
        "connections": float_connection_count,
        "density_requested": density,
        "active_pre_ratio": active_ratio,
        "steps": steps,
        "step_ms": elapsed * 1000.0 / steps,
        "layer_step_ms": layer_elapsed * 1000.0 / steps,
        "active_connections_per_step_approx": (
            float_connection_count * active_ratio
        ),
        "inactive_presynaptic_neurons_skipped": total_skipped,
        "network_array_bytes": synapse.memory_usage_bytes,
        "float_weight_bytes": float_weight_bytes,
        "stored_weight_bytes": synapse.weights.nbytes,
        "working_set_delta_bytes": max(0, rss - baseline_memory),
        "quantized_top1_agreement": agreement,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--active-ratio", type=float, default=0.01)
    parser.add_argument("--quantized", action="store_true")
    args = parser.parse_args()
    if args.neurons <= 0 or args.steps <= 0:
        parser.error("--neurons y --steps deben ser mayores que cero")
    if not 0.0 < args.density <= 1.0:
        parser.error("--density debe estar en (0, 1]")
    if not 0.0 < args.active_ratio <= 1.0:
        parser.error("--active-ratio debe estar en (0, 1]")
    for name, value in run_benchmark(
        args.neurons,
        args.density,
        args.steps,
        args.active_ratio,
        args.quantized,
    ).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
