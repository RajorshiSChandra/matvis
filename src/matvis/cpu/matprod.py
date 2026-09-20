"""CPU-based source-summing operations."""

import numpy as np

from ..core.matprod import MatProd


class CPUMatMul(MatProd):
    """Use simple numpy.dot to perform the source-summing operation."""

    def compute(
        self, z: np.ndarray, out: np.ndarray, sgn: np.ndarray | None = None
    ) -> np.ndarray:
        """Perform the source-summing operation for a single time and chunk.

        Parameters
        ----------
        z
            Complex integrand. Shape=(Nfeed, Nant, Nax, Nsrc).
        out
            Output array, shaped as (Nfeed, Nfeed, Npairs).
        sgn
            Optional per-column source-sign vector, length ``z.shape[-1]``, for
            skies containing negative brightness. ``None`` (all-non-negative sky)
            uses the legacy path.
        """
        # sgn carries the flux sign in ONE factor only: V = Z* diag(sgn) Z^T,
        # so each source contributes sign(I)*|I|/2 = I/2 exactly.
        zs = z if sgn is None else z * sgn
        v = z.conj().dot(zs.T)

        # Separate feed/ant axes to make indexing easier
        v.shape = (self.nant, self.nfeed, self.nant, self.nfeed)
        v = v.transpose((0, 2, 3, 1))  # transpose always returns a view

        if self.all_pairs:
            out[:] = v.reshape((self.nant * self.nant, self.nfeed, self.nfeed))
        else:
            out[:] = v[self.ant1_idx, self.ant2_idx]

        return out


class CPUVectorDot(MatProd):
    """Use a loop over specific pairs, performing a vdot over the source axis."""

    def compute(
        self, z: np.ndarray, out: np.ndarray, sgn: np.ndarray | None = None
    ) -> np.ndarray:
        """Perform the source-summing operation for a single time and chunk.

        Parameters
        ----------
        z
            Complex integrand. Shape=(Nfeed, Nant, Nax, Nsrc).
        out
            Output array, shaped as (Nfeed, Nfeed, Npairs).
        sgn
            Optional per-column source-sign vector, length ``z.shape[-1]``, for
            skies containing negative brightness. ``None`` (all-non-negative sky)
            uses the legacy path.
        """
        z = z.reshape((self.nant, self.nfeed, -1))
        # Sign goes on the unconjugated factor (sgn is real, so either side is
        # mathematically identical): V = Z* diag(sgn) Z^T.
        zs = z if sgn is None else z * sgn

        for i, (ai, aj) in enumerate(self.antpairs):
            out[i] = zs[aj].dot(z[ai].conj().T)  # dot(zs[aj].T)

        return out
