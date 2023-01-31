from typing import Dict
import torch
import numpy as np
import scipy.sparse

from dataclasses import dataclass


def is_scipysparse(value):
    return isinstance(value, scipy.sparse.spmatrix)


def is_sparse(value):
    return isinstance(value, Sparse)

class Sparse:
    pass


@dataclass
class CSR(Sparse):
    """
    CSR sparse tensor
    We don't use the torch variant, because this is not supported on CUDA (and we would not use any of its functions anyway)
    """

    indptr: torch.tensor
    indices: torch.tensor
    values: torch.tensor
    shape: torch.Size

    @classmethod
    def from_scipy_csr(cls, scipy_csr, device="cpu", dtype=None):
        # will keep original numpy data type unless specified
        if dtype is None:
            values = torch.tensor(scipy_csr.data, device=device)
        else:
            values = torch.tensor(scipy_csr.data, device=device, dtype=dtype)
        return cls(
            indices=torch.tensor(scipy_csr.indices, device=device, dtype=torch.long),
            indptr=torch.tensor(scipy_csr.indptr, device=device, dtype=torch.long),
            values=values,
            shape=scipy_csr.shape,
        )


@dataclass
class COOMatrix(Sparse):
    """
    COOMatrix sparse matrix

    This implementation mainly cachges the row_switch and mapping dictionaries, which speed up subsampling in the first (row) dimension
    """

    row: torch.tensor
    col: torch.tensor
    values: torch.tensor
    shape: torch.Size
    row_switch = None
    mapping: Dict = None

    ndim = 2

    def populate_row_switch(self):
        device = self.row.device
        self.row_switch = torch.cat(
            (
                torch.tensor([0], device=device),
                torch.where(self.row[1:] - self.row[:-1])[0] + 1,
                torch.tensor([len(self.values)], device=device),
            )
        )
        assert (
            len(self.row_switch) == self.shape[0] + 1
        ), "There are probably some rows with only zeros, this is not supported"

    def populate_mapping(self):
        # this maps each row to its value and column indices
        # this allows us to go subset in the row dimension very quickly, by just extracting the relevant rows from this dictionary
        # at the cost of having to keep these mappings in memory
        device = self.row.device
        self.populate_row_switch()
        self.mapping = {
            row: torch.arange(
                self.row_switch[row], self.row_switch[row + 1], device=device
            )
            for row in range(self.shape[0])
        }

    def dense_slice(self, ix):
        """
        Slice (in the rows) and return a dense tensor. ix should be a slice.
        This is a bit faster, at the cost of requiring continuity in the subsampling
        Nonetheless, `dense_subset()` implementation is already quite fast, so this restriction is not necessary
        """
        return self.dense_subset(ix)

    def dense_subset(self, ix):
        """
        Subset (in the rows) and return a dense tensor. ix should be an np array.
        """

        if torch.is_tensor(ix):
            ix = ix.cpu().numpy()  #! This can be a major slow down
        elif isinstance(ix, slice):
            ix = np.arange(ix.start, ix.stop)
        elif isinstance(ix, np.ndarray):
            pass
        else:
            raise ValueError("Cannot take dense_subset of " + str(ix.__class__) + ", requires a tensor, slice or numpy.ndarray")

        device = self.row.device
        ix_sparse = torch.cat([self.mapping[row] for row in ix])
        newrow_idx = torch.arange(len(ix), device=device).repeat_interleave(
            self.row_switch[ix + 1] - self.row_switch[ix]
        )
        newcol_idx = self.col[ix_sparse]

        subsetted = torch.zeros(
            (len(ix), self.shape[1]), device=device, dtype=self.values.dtype
        )
        subsetted[newrow_idx, newcol_idx] = self.values[ix_sparse]
        return subsetted

    def dense(self):
        device = self.row.device
        subsetted = torch.zeros(self.shape, device=device, dtype=self.values.dtype)
        subsetted[self.row, self.col] = self.values
        return subsetted

    @classmethod
    def from_numpy_array(cls, ndarray, device="cpu", dtype=None):
        scipy_coo = scipy.sparse.coo_matrix(ndarray)
        return cls.from_scipy_coo(scipy_coo, device, dtype)

    @classmethod
    def from_scipy_coo(cls, scipy_coo, device="cpu", dtype=None):
        # will keep original numpy data type unless specified
        if dtype is None:
            values = torch.tensor(scipy_coo.data, device=device)
        else:
            values = torch.tensor(scipy_coo.data, device=device, dtype=dtype)
        return cls(
            row=torch.tensor(scipy_coo.row, device=device, dtype=torch.long),
            col=torch.tensor(scipy_coo.col, device=device, dtype=torch.long),
            values=values,
            shape=scipy_coo.shape,
        )

    @classmethod
    def from_scipy_csr(cls, scipy_csr, device="cpu", dtype=None):
        return cls.from_scipy_coo(scipy_csr.tocoo(), device=device, dtype=dtype)  # 😎

    @classmethod
    def from_scipy(cls, scipy_sparse, device="cpu", dtype=None):
        if isinstance(scipy_sparse, scipy.sparse.csr_matrix):
            return cls.from_scipy_csr(scipy_sparse, device=device, dtype=dtype)
        elif isinstance(scipy_sparse, scipy.sparse.coo_matrix):
            return cls.from_scipy_coo(scipy_sparse, device=device, dtype=dtype)
        else:
            raise NotImplementedError()

    def to_np(self):
        return self.dense().numpy()

    def to_scipy_coo(self):
        return scipy.sparse.coo_matrix(
            (self.values.numpy(), (self.row.numpy(), self.col.numpy())),
            shape=self.shape,
        )

    def to_scipy_csr(self):
        return self.to_scipy_coo().tocsr()

    def to_coo(self):
        return self

    def to(self, device):
        self.row = self.row.to(device)
        self.col = self.col.to(device)
        self.values = self.values.to(device)
        if self.row_switch is not None:
            self.row_switch = self.row_switch.to(device)
        if self.mapping is not None:
            self.mapping = {
                key: value.to(device) for key, value in self.mapping.items()
            }
        return self

    def __getitem__(self, ix):
        x = self.__class__.from_scipy_csr(self.to_scipy_csr()[:, ix[1]])
        x.populate_mapping()
        x.populate_row_switch()
        return x

    def __repr__(self):
        return f"{self.__class__}"
