from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import numpy as np

from tinygp import GaussianProcess
from tinygp.kernels import quasisep


jax.config.update("jax_enable_x64", True)


def build_kernel(name: str):
    if name == "matern32":
        return quasisep.Matern32(sigma=1.8, scale=1.5)
    if name == "matern52":
        return quasisep.Matern52(sigma=1.8, scale=1.5)
    if name == "exp":
        return quasisep.Exp(sigma=1.8, scale=1.5)
    if name == "celerite":
        return quasisep.Celerite(1.1, 0.8, 0.9, 0.1)
    raise ValueError(f"Unknown kernel '{name}'")


def make_data(n: int, seed: int) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    x = jnp.sort(jnp.asarray(rng.uniform(-3.0, 3.0, size=n)))
    y = jnp.sin(x)
    return x, y


def time_call(fn, *, repeats: int, warmup: int = 1) -> dict[str, float]:
    compile_start = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    compile_and_first = time.perf_counter() - compile_start

    for _ in range(max(0, warmup - 1)):
        out = fn()
        jax.block_until_ready(out)

    runtimes = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        runtimes.append(time.perf_counter() - start)

    return {
        "compile_plus_first_s": compile_and_first,
        "steady_mean_s": float(np.mean(runtimes)),
        "steady_min_s": float(np.min(runtimes)),
        "steady_max_s": float(np.max(runtimes)),
    }


def run_case(kernel_name: str, n: int, repeats: int, warmup: int) -> dict[str, object]:
    kernel = build_kernel(kernel_name)
    x, y = make_data(n, seed=1234 + n)
    gp = GaussianProcess(kernel, x, diag=0.1)
    matrix = kernel.to_symm_qsm(x) + gp.noise.to_qsm()
    factor = matrix.cholesky()

    results = {
        "kernel": kernel_name,
        "n": n,
        "state_dim": int(kernel.design_matrix().shape[0]),
        "to_symm_qsm": time_call(lambda: kernel.to_symm_qsm(x), repeats=repeats, warmup=warmup),
        "cholesky": time_call(lambda: matrix.cholesky(), repeats=repeats, warmup=warmup),
        "solve_triangular": time_call(
            lambda: factor.solve(y), repeats=repeats, warmup=warmup
        ),
        "log_probability": time_call(
            lambda: gp.log_probability(y), repeats=repeats, warmup=warmup
        ),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tinygp quasisep CPU hot paths.")
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["matern32", "matern52", "exp", "celerite"],
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = {
        "python": sys.version,
        "jax_version": jax.__version__,
        "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
        "devices": [str(d) for d in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "results": [],
    }

    for kernel_name in args.kernels:
        for n in args.sizes:
            payload["results"].append(
                run_case(kernel_name, n=n, repeats=args.repeats, warmup=args.warmup)
            )

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"jax={payload['jax_version']} jaxlib={payload['jaxlib_version']}")
    print(f"devices={payload['devices']}")
    print(f"XLA_FLAGS={payload['xla_flags']!r}")
    for row in payload["results"]:
        print(
            f"{row['kernel']:>8} n={row['n']:>5} state={row['state_dim']:>2} "
            f"to_symm={row['to_symm_qsm']['steady_mean_s']:.6f}s "
            f"chol={row['cholesky']['steady_mean_s']:.6f}s "
            f"solve={row['solve_triangular']['steady_mean_s']:.6f}s "
            f"logp={row['log_probability']['steady_mean_s']:.6f}s"
        )


if __name__ == "__main__":
    main()
