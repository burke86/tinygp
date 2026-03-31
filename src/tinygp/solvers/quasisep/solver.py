from __future__ import annotations

__all__ = ["QuasisepSolver"]

from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from tinygp.helpers import JAXArray
from tinygp.kernels.base import Kernel
from tinygp.noise import Noise
from tinygp.solvers.quasisep.core import DiagQSM, LowerTriQSM, SymmQSM
from tinygp.solvers.solver import Solver


@jax.jit
def _add_noise(matrix: SymmQSM, noise_qsm: SymmQSM | DiagQSM) -> SymmQSM:
    result = matrix + noise_qsm
    assert isinstance(result, SymmQSM)
    return result


@jax.jit
def _factorize(matrix: SymmQSM) -> LowerTriQSM:
    return matrix.cholesky()


@jax.jit
def _normalization_from_diag(diag: JAXArray) -> JAXArray:
    return jnp.sum(jnp.log(diag)) + 0.5 * diag.shape[0] * np.log(2 * np.pi)


@jax.jit
def _centered_log_probability(factor: LowerTriQSM, y: JAXArray, norm: JAXArray) -> JAXArray:
    y = jnp.reshape(y, (y.shape[0], -1))

    def impl(carry, data):  # type: ignore
        fn, quad = carry
        ((cn,), (pn, wn, an)), yn = data
        xn = (yn - pn @ fn) / cn
        fn = an @ fn + jnp.outer(wn, xn)
        quad = quad + jnp.sum(jnp.square(xn))
        return (fn, quad), None

    init_dtype = jnp.result_type(factor.lower.q.dtype, y.dtype)
    init_f = jnp.zeros((factor.lower.q.shape[1], y.shape[1]), dtype=init_dtype)
    init_q = jnp.zeros((), dtype=jnp.result_type(init_dtype, factor.diag.d.dtype))
    (_, quad), _ = jax.lax.scan(impl, (init_f, init_q), (factor, y))
    loglike = -0.5 * quad - norm
    return jnp.where(jnp.isfinite(loglike), loglike, -jnp.inf)


class QuasisepSolver(Solver):
    """A scalable solver that uses quasiseparable matrices

    Take a look at the documentation for the :ref:`api-solvers-quasisep`, for
    more technical details.

    You generally won't instantiate this object directly but, if you do, you'll
    probably want to use the :func:`QuasisepSolver.init` method instead of the
    usual constructor.
    """

    X: JAXArray
    matrix: SymmQSM
    factor: LowerTriQSM
    normalization_value: JAXArray

    def __init__(
        self,
        kernel: Kernel,
        X: JAXArray,
        noise: Noise,
        *,
        covariance: Any | None = None,
        assume_sorted: bool = False,
    ):
        """Build a :class:`QuasisepSolver` for a given kernel and coordinates

        Args:
            kernel: The kernel function. This must be an instance of a subclass
                of :class:`tinygp.kernels.quasisep.Quasisep`.
            X: The input coordinates.
            noise: The noise model for the process.
            covariance: Optionally, a pre-computed
                :class:`tinygp.solvers.quasisep.core.QSM` with the covariance
                matrix.
            assume_sorted: If ``True``, assume that the input coordinates are
                sorted. If ``False``, check that they are sorted and throw an
                error if they are not. This can introduce a runtime overhead,
                and you can pass ``assume_sorted=True`` to get the best
                performance.
        """
        from tinygp.kernels.quasisep import Quasisep

        if covariance is None:
            if TYPE_CHECKING:
                assert isinstance(kernel, Quasisep)
            if not assume_sorted:
                jax.debug.callback(_check_sorted, kernel.coord_to_sortable(X))
            matrix = _add_noise(kernel.to_symm_qsm(X), noise.to_qsm())
        else:
            if TYPE_CHECKING:
                assert isinstance(covariance, SymmQSM)
            matrix = covariance
        self.X = X
        self.matrix = matrix
        self.factor = _factorize(matrix)
        self.normalization_value = _normalization_from_diag(self.factor.diag.d)

    def variance(self) -> JAXArray:
        return self.matrix.diag.d

    def covariance(self) -> JAXArray:
        return self.matrix.to_dense()

    @jax.jit
    def normalization(self) -> JAXArray:
        return self.normalization_value

    @jax.jit
    def centered_log_probability(self, y: JAXArray) -> JAXArray:
        return _centered_log_probability(self.factor, y, self.normalization_value)

    @partial(jax.jit, static_argnames=("transpose",))
    def solve_triangular(self, y: JAXArray, *, transpose: bool = False) -> JAXArray:
        if transpose:
            return self.factor.transpose().solve(y)
        else:
            return self.factor.solve(y)

    @jax.jit
    def dot_triangular(self, y: JAXArray) -> JAXArray:
        return self.factor @ y

    def condition(self, kernel: Kernel, X_test: JAXArray | None, noise: Noise) -> Any:
        """Compute the covariance matrix for a conditional GP

        In the case where the prediction is made at the input coordinates with a
        :class:`tinygp.kernels.quasisep.Quasisep` kernel, this will return the
        quasiseparable representation of the conditional matrix. Otherwise, it
        will use scalable methods where possible, but return a dense
        representation of the covariance, so be careful when predicting at a
        large number of test points!

        Args:
            kernel: The kernel for the covariance between the observed and
                predicted data.
            X_test: The coordinates of the predicted points. Defaults to the
                input coordinates.
            noise: The noise model for the predicted process.
        """
        from tinygp.kernels.quasisep import Quasisep

        # We can easily compute the conditional as a QSM in the special case
        # where we are predicting at the input coordinates and a Quasisep kernel
        if X_test is None and isinstance(kernel, Quasisep):
            M = kernel.to_symm_qsm(self.X)
            delta = (self.factor.inv() @ M).gram()
            M += noise.to_qsm()
            return M - delta

        # Otherwise fall back on the slow method for now :(
        if X_test is None:
            Kss = Ks = kernel(self.X, self.X)
        else:
            Kss = kernel(X_test, X_test)
            Ks = kernel(self.X, X_test)

        A = self.solve_triangular(Ks)
        return Kss - A.transpose() @ A


def _check_sorted(X: JAXArray) -> None:
    if np.any(np.diff(X) < 0.0):
        raise ValueError(
            "Input coordinates must be sorted in order to use the QuasisepSolver"
        )
