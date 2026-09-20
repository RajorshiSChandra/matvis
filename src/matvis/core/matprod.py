"""Base class for performing the source-summing operation."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .._utils import get_dtypes


class MatProd(ABC):
    """
    Abstract base class for performing the source-summing operation.

    Parameters
    ----------
    nchunks
        Number of chunks to split the sources into.
    nfeed
        Number of feeds.
    nant
        Number of antennas.
    antpairs
        The antenna pairs to sum over. If None, all pairs are used.
    precision
        The precision of the data (1 or 2).
    """

    def __init__(
        self,
        nchunks: int,
        nfeed: int,
        nant: int,
        antpairs: np.ndarray | None,
        precision=1,
    ):
        if antpairs is None:
            self.all_pairs = True
            self.antpairs = np.array([(i, j) for i in range(nant) for j in range(nant)])
        else:
            self.all_pairs = False
            self.antpairs = antpairs

        self.nchunks = nchunks
        self.nfeed = nfeed
        self.nant = nant

        self.npairs = len(self.antpairs)
        self.ctype = get_dtypes(precision)[1]

        self.ant1_idx = self.antpairs[:, 0]
        self.ant2_idx = self.antpairs[:, 1]

    def allocate_vis(self):
        """Allocate memory for the visibilities.

        The shape of the visibilities must have a first axis of length nchunks,
        but then can be arbitrary shaped after that, so long as it is consistently
        used throughout the class.
        """
        self.vis = np.full(
            (self.nchunks, self.npairs, self.nfeed, self.nfeed), 0.0, dtype=self.ctype
        )

    def setup(self):
        """Setup the memory for the object."""
        self.allocate_vis()

    @abstractmethod
    def compute(self, z: np.ndarray, out: np.ndarray, sgn: np.ndarray | None = None):
        """
        Perform the source-summing operation for a single time and chunk.

        Parameters
        ----------
        z
            Complex integrand. Shape=(Nant, Nfeed, Nax, Nsrc).
        out
            Output array, shaped like the visibilities set in `allocate_vis`, but
            without the first chunk axis.
        sgn
            Optional per-column source-sign vector, length ``z.shape[-1]``, for
            skies containing negative brightness (V = Z* diag(sgn) Z^T). ``None``
            means an all-non-negative sky and must reproduce the legacy path
            exactly.
        """

    def __call__(
        self, z: np.ndarray, chunk: int, sgn: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Perform the source-summing operation for a single time and chunk.

        Parameters
        ----------
        z
            Complex integrand. Shape=(Nant, Nfeed, Nax, Nsrc).
        chunk
            The chunk index.
        sgn
            Optional per-column source-sign vector (see :meth:`compute`).

        Returns
        -------
        out
            The output array, shaped like the visibilities set in `allocate_vis`, but
            without the first chunk axis.
        """
        if sgn is None:
            # Legacy call form: keeps subclasses with the pre-sgn compute()
            # signature (e.g. GPU backends, external MatProd subclasses) working.
            self.compute(z, out=self.vis[chunk])
        else:
            self.compute(z, out=self.vis[chunk], sgn=sgn)
        return self.vis[chunk]

    def sum_chunks(self, out: np.ndarray):
        """
        Sum the chunks into the output array.

        Parameters
        ----------
        out
            The output visibilities, with shape (Nfeed, Nfeed, Npairs).
        """
        if self.nchunks == 1:
            out[:] = self.vis[0]
        else:
            self.vis.sum(axis=0, out=out)
