from dataclasses import dataclass
from typing import Dict

import numpy as np
import scipy.sparse
import torch


import itertools
import pathlib
import pickle

import pandas as pd
import zarr


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
    We don't use the torch variant, because this is not
    supported on CUDA (and we would not use any of its functions anyway)
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

    Additional CSR-like data can be stored to speed up subsampling
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
        # this allows us to go subset in the row dimension very quickly,
        # by just extracting the relevant rows from this dictionary
        # at the cost of having to keep these mappings in memory
        device = self.row.device
        self.populate_row_switch()
        self.mapping = {
            row: torch.arange(self.row_switch[row], self.row_switch[row + 1], device=device)
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
            raise ValueError(
                "Cannot take dense_subset of " + str(ix.__class__) + ", requires a tensor, slice or numpy.ndarray"
            )

        device = self.row.device
        ix_sparse = torch.cat([self.mapping[row] for row in ix])
        newrow_idx = torch.arange(len(ix), device=device).repeat_interleave(
            self.row_switch[ix + 1] - self.row_switch[ix]
        )
        newcol_idx = self.col[ix_sparse]

        subsetted = torch.zeros((len(ix), self.shape[1]), device=device, dtype=self.values.dtype)
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
            self.mapping = {key: value.to(device) for key, value in self.mapping.items()}
        return self

    def __getitem__(self, ix):
        x = self.__class__.from_scipy_csr(self.to_scipy_csr()[:, ix[1]])
        # x.populate_mapping()
        # x.populate_row_switch()
        return x

    def __repr__(self):
        return f"{self.__class__}"


def _create_variable(path, variable, dimensions, sparse, dtype, coords_pointed, fill_value=0.0):
    assert isinstance(dimensions, tuple)

    if sparse:
        zarr.create(
            store=path / (variable + ".data.zarr"),
            overwrite=True,
            chunks=(1000,),
            shape=(0,),
            dtype=dtype,
        )
        zarr.create(
            store=path / (variable + ".indices.zarr"),
            overwrite=True,
            chunks=(1000,),
            shape=(0,),
            dtype=np.int32,
        )

        shape = [len(coords_pointed[dimension]) for dimension in dimensions if dimension in coords_pointed]
        zarr.create(
            store=path / (variable + ".indptr.zarr"),
            overwrite=True,
            chunks=(1000, 2),
            shape=(np.prod(shape) + 1, 2),
            dtype=np.int32,
        )
    else:
        shape = [len(coords_pointed[dimension]) for dimension in dimensions if dimension in coords_pointed]
        zarr.create(
            store=path / (variable + ".zarr"),
            overwrite=True,
            chunks=(1000,),
            shape=tuple(shape),
            dtype=dtype,
            fill_value=fill_value,
        )


def _load_variable(path, variable, dimensions, sparse, coords_pointed, coords_fixed):
    if sparse:
        return SparseDatarray(
            path,
            variable,
            {g: coords_pointed[g] for g in dimensions if g in coords_pointed},
            {g: coords_fixed[g] for g in dimensions if g in coords_fixed},
        )
    else:
        return Datarray(
            path,
            variable,
            {g: coords_pointed[g] for g in dimensions if g in coords_pointed},
            {g: coords_fixed[g] for g in dimensions if g in coords_fixed},
        )


class SparseDataset:
    def __init__(self, path, coords_pointed, coords_fixed, variables):
        self.coords_pointed = coords_pointed
        self.coords_fixed = coords_fixed
        self.variables = variables
        self.path = path

        self._arrays = {}
        for variable, variable_info in variables.items():
            dimensions = variable_info["dimensions"]
            sparse = variable_info["sparse"] if "sparse" in variable_info else True

            self._arrays[variable] = _load_variable(
                path,
                variable,
                dimensions,
                sparse,
                coords_pointed,
                coords_fixed,
            )

    @classmethod
    def create(cls, path, variables, coords_pointed, coords_fixed):
        assert isinstance(variables, dict)

        for k, v in coords_pointed.items():
            assert isinstance(v, pd.Index)

        for k, v in coords_fixed.items():
            assert isinstance(v, pd.Index)

        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        for variable, variable_info in variables.items():
            fill_value = variable_info["fill_value"] if "fill_value" in variable_info else 0.0
            _create_variable(
                path,
                variable,
                dimensions=variable_info["dimensions"],
                sparse=True if "sparse" not in variable_info else variable_info["sparse"],
                coords_pointed=coords_pointed,
                dtype=np.float32 if "dtype" not in variable_info else variable_info["dtype"],
                fill_value=fill_value,
            )
        pickle.dump(coords_pointed, open(path / "coords_pointed.pkl", "wb"))
        pickle.dump(coords_fixed, open(path / "coords_fixed.pkl", "wb"))
        pickle.dump(variables, open(path / "variables.pkl", "wb"))

        self = cls(path, coords_pointed, coords_fixed, variables)

        return self

    def create_variable(self, variable, dimensions, sparse=True, dtype=np.float32, fill_value=0.0):
        if variable in self.variables:
            raise ValueError("Variable already exists")
        self.variables[variable] = dict(dimensions=dimensions, sparse=sparse, dtype=dtype)
        _create_variable(
            self.path,
            variable,
            dimensions=dimensions,
            sparse=sparse,
            coords_pointed=self.coords_pointed,
            dtype=dtype,
            fill_value=fill_value,
        )
        pickle.dump(self.variables, open(self.path / "variables.pkl", "wb"))
        self._arrays[variable] = _load_variable(
            self.path,
            variable,
            dimensions,
            sparse,
            self.coords_pointed,
            self.coords_fixed,
        )

    @classmethod
    def open(cls, path):
        path = pathlib.Path(path)
        variables = pickle.load(open(path / "variables.pkl", "rb"))
        coords_pointed = pickle.load(open(path / "coords_pointed.pkl", "rb"))
        coords_fixed = pickle.load(open(path / "coords_fixed.pkl", "rb"))

        return cls(path, coords_pointed, coords_fixed, variables)

    @property
    def coords(self):
        return self.coords_pointed

    def __getitem__(self, k):
        return self._arrays[k]

    def sel_xr(self, k=None, variables=None):
        import xarray as xr

        if variables is None:
            variables = self._arrays.keys()

        v = {variable: self._arrays[variable].sel_xr(k) for variable in variables}
        return xr.Dataset(v)

    # html repr
    def _repr_html_(self):
        import pandas as pd

        df_variables = pd.DataFrame(
            [
                [variable, variable_info["dimensions"], variable_info["sparse"] if "sparse" in variable_info else True]
                for variable, variable_info in self.variables.items()
            ],
            columns=["variable", "dimensions", "sparse"],
        ).set_index("variable")
        df_dimensions = pd.DataFrame(
            [[dimension, len(coord), False] for dimension, coord in self.coords_pointed.items()]
            + [[dimension, len(coord), True] for dimension, coord in self.coords_fixed.items()],
            columns=["dimension", "length", "fixed"],
        ).set_index("dimension")
        return "Variables:\n" + df_variables._repr_html_() + "\nDimensions:\n" + df_dimensions._repr_html_()

    def __contains__(self, k):
        return k in self._arrays


def process_key(k, coords_pointed):
    if k is None:
        k = tuple([slice(None)] * len(coords_pointed))
    elif not isinstance(k, tuple):
        k = (k,)
    if len(k) < len(coords_pointed):
        k = k + tuple([slice(None)] * (len(coords_pointed) - len(k)))
    return k


def key_to_ix(k, coords_pointed, unwrap=False):
    ix = []
    for coord, ki in zip(coords_pointed.values(), k):
        if isinstance(ki, slice):
            start = coord.get_indexer(ki.start) if ki.start is not None else None
            stop = coord.get_indexer(ki.stop) if ki.stop is not None else None
            if unwrap:
                if start is None:
                    start = 0
                if stop is None:
                    stop = len(coord)
                ix.append(np.arange(start, stop))
            else:
                ix.append(slice(start, stop, ki.step))
        elif isinstance(ki, (pd.Index, pd.Series, np.ndarray, list)):
            ix.append(coord.get_indexer(ki))
        else:
            ix.append(coord.get_loc(ki))
    return ix


class Datarray:
    _reader = None

    def __init__(self, path, variable, coords_pointed, coords_fixed):
        self.path = path
        self.variable = variable
        self.coords_pointed = coords_pointed
        self.shape_pointed = tuple([len(self.coords_pointed[g]) for g in self.coords_pointed])
        self.coords_fixed = coords_fixed
        self.shape_fixed = tuple([len(self.coords_fixed[g]) for g in self.coords_fixed])

    def _read(self):
        if self._reader is None:
            self._reader = zarr.open(self.path / (self.variable + ".zarr"), mode="r")
        return self._reader

    def __getitem__(self, k):
        k = process_key(k, self.coords_pointed)
        ix = key_to_ix(k, self.coords_pointed)
        return self._read()[tuple(ix)]

    def sel_xr(self, k=None):
        k = process_key(k, self.coords_pointed)
        v = self.__getitem__(k)
        coords = key_to_coords(k, self.coords_pointed, self.coords_fixed)
        import xarray as xr

        return xr.DataArray(v, coords=coords)

    def __setitem__(self, k, v):
        k = process_key(k, self.coords_pointed)
        ix = key_to_ix(k, self.coords_pointed)

        data = zarr.open(self.path / (self.variable + ".zarr"), mode="r+")
        data[tuple(ix)] = v


class SparseDatarray:
    def __init__(self, path, variable, coords_pointed, coords_fixed):
        self.path = path
        self.variable = variable
        self.coords_pointed = coords_pointed
        self.shape_pointed = tuple([len(self.coords_pointed[g]) for g in self.coords_pointed])
        self.coords_fixed = coords_fixed
        self.shape_fixed = tuple([len(self.coords_fixed[g]) for g in self.coords_fixed])

    def __setitem__(self, k, v):
        k = process_key(k, self.coords_pointed)

        if not isinstance(v, np.ndarray):
            v = np.array(v)

        assert v.ndim == len(self.shape_fixed), (v.ndim, len(self.shape_fixed))
        assert v.shape == self.shape_fixed, (v.shape, self.shape_fixed)

        data = zarr.open(self.path / (self.variable + ".data.zarr"), mode="a")
        indices = zarr.open(self.path / (self.variable + ".indices.zarr"), mode="a")
        indptr = zarr.open(self.path / (self.variable + ".indptr.zarr"), mode="a")

        v_nonzero = np.nonzero(v.flatten())[0]
        v_data = v.flatten()[v_nonzero]
        v_indices = v_nonzero

        if len(v_data) == 0:
            return

        data_start, data_end = len(data), len(data) + len(v_data)
        data.append(v_data)
        indices.append(v_indices)

        indptr_key = np.ravel_multi_index(
            [coord.get_loc(ki) for coord, ki in zip(self.coords_pointed.values(), k)], self.shape_pointed
        )
        indptr[indptr_key] = [data_start, data_end]

    def __getitem__(self, k):
        k = process_key(k, self.coords_pointed)

        indptr_key = [coord.get_loc(ki) for coord, ki in zip(self.coords_pointed.values(), k)]

        return self._get_ix(indptr_key)

    def _get_ix(self, indptr_key):
        indptr = self._read_indptr()
        data = self._read_data()
        indices = self._read_indices()

        data_key, data_key_end = indptr[np.ravel_multi_index(indptr_key, self.shape_pointed)][:]

        data = data[data_key:data_key_end]
        indices = indices[data_key:data_key_end]

        value = np.zeros(self.shape_fixed, dtype=data.dtype)
        if len(data) > 0:
            if value.ndim > 0:
                value[np.unravel_index(indices, value.shape)] = data
            else:
                return data[0]
        return value

    def _read_indptr(self):
        return zarr.open(self.path / (self.variable + ".indptr.zarr"), mode="r")

    def _read_data(self):
        return zarr.open(self.path / (self.variable + ".data.zarr"), mode="r")

    def _read_indices(self):
        return zarr.open(self.path / (self.variable + ".indices.zarr"), mode="r")

    def sel(self, k=None):
        k = process_key(k, self.coords_pointed)
        ix = key_to_ix(k, self.coords_pointed, unwrap=True)

        v = []
        for ksubset in itertools.product(*[[i] if isinstance(i, int) else i for i in ix]):
            v.append(self._get_ix(ksubset))
        v = np.stack(v).reshape(tuple([len(i) for i in ix if not isinstance(i, int)] + list(self.shape_fixed)))
        return v

    def sel_xr(self, k=None):
        k = process_key(k, self.coords_pointed)
        v = self.sel(k)
        coords = key_to_coords(k, self.coords_pointed, self.coords_fixed)
        import xarray as xr

        return xr.DataArray(v, coords=coords)


def key_to_coords(k, coords_pointed, coords_fixed):
    coords = {}
    for (coord_name, coord), ki in zip(coords_pointed.items(), k):
        if isinstance(ki, slice):
            start = coord.get_indexer(ki.start) if ki.start is not None else None
            stop = coord.get_indexer(ki.stop) if ki.stop is not None else None
            coords[coord.name] = coord[start:stop]
        elif isinstance(ki, (pd.Index, pd.Series, np.ndarray, list)):
            coords[coord.name] = coord[coord.get_indexer(ki)]
    return {**coords, **coords_fixed}


class SparseDataset2:
    def __init__(self, path, coords_pointed, coords_fixed, variables):
        self.coords_pointed = coords_pointed
        self.coords_fixed = coords_fixed
        self.variables = variables
        self.path = path

        self._arrays = {}
        for variable, variable_info in variables.items():
            dimensions = variable_info["dimensions"]
            sparse = variable_info["sparse"] if "sparse" in variable_info else True

            if sparse:
                self._arrays[variable] = SparseDatarray2(
                    path,
                    variable,
                    {g: coords_pointed[g] for g in dimensions if g in coords_pointed},
                    {g: coords_fixed[g] for g in dimensions if g in coords_fixed},
                )
            else:
                self._arrays[variable] = Datarray2(
                    path,
                    variable,
                    {g: coords_pointed[g] for g in dimensions if g in coords_pointed},
                    {g: coords_fixed[g] for g in dimensions if g in coords_fixed},
                )

    @classmethod
    def create(cls, path, variables, coords_pointed, coords_fixed):
        assert isinstance(variables, dict)

        for k, v in coords_pointed.items():
            assert isinstance(v, pd.Index)

        for k, v in coords_fixed.items():
            assert isinstance(v, pd.Index)

        path = pathlib.Path(path)

        self = cls(path, coords_pointed, coords_fixed, variables)

        return self

    @classmethod
    def open(cls, path):
        path = pathlib.Path(path)
        variables = pickle.load(open(path / "variables.pkl", "rb"))
        coords_pointed = pickle.load(open(path / "coords_pointed.pkl", "rb"))
        coords_fixed = pickle.load(open(path / "coords_fixed.pkl", "rb"))

        return cls(path, coords_pointed, coords_fixed, variables)

    @property
    def coords(self):
        return self.coords_pointed

    def __getitem__(self, k):
        return self._arrays[k]

    def sel_xr(self, k=None, variables=None):
        import xarray as xr

        if variables is None:
            variables = self._arrays.keys()

        v = {variable: self._arrays[variable].sel_xr(k) for variable in variables}
        return xr.Dataset(v)


class Datarray2:
    def __init__(self, path, variable, coords_pointed, coords_fixed):
        self.path = path
        self.variable = variable
        self.coords_pointed = coords_pointed
        self.shape_pointed = tuple([len(self.coords_pointed[g]) for g in self.coords_pointed])
        self.coords_fixed = coords_fixed
        self.shape_fixed = tuple([len(self.coords_fixed[g]) for g in self.coords_fixed])

    def __getitem__(self, k):
        k = process_key(k, self.coords_pointed)
        path = self.path / self.variable / ("_".join(k) + ".zarr")
        if not path.exists():
            return np.zeros(self.shape_fixed)
        return zarr.open(self.path / self.variable / ("_".join(k) + ".zarr"), mode="r")

    def sel_xr(self, k=None):
        k = process_key(k, self.coords_pointed)
        v = self.__getitem__(k)
        coords = key_to_coords(k, self.coords_pointed, self.coords_fixed)
        import xarray as xr

        return xr.DataArray(v, coords=coords)

    def _get_key_path(self, k):
        return self.path / self.variable / ("_".join(k) + ".zarr")

    def __setitem__(self, k, v):
        k = process_key(k, self.coords_pointed)

        zarr.save(self._get_key_path(k), v)


class SparseDatarray2:
    def __init__(self, path, variable, coords_pointed, coords_fixed):
        self.path = path
        self.variable = variable
        self.coords_pointed = coords_pointed
        self.shape_pointed = tuple([len(self.coords_pointed[g]) for g in self.coords_pointed])
        self.coords_fixed = coords_fixed
        self.shape_fixed = tuple([len(self.coords_fixed[g]) for g in self.coords_fixed])

    def _get_key_path_data(self, k):
        return self.path / self.variable / ("_".join(k) + ".data.zarr")

    def _get_key_path_indices(self, k):
        return self.path / self.variable / ("_".join(k) + ".indices.zarr")

    def __setitem__(self, k, v):
        k = process_key(k, self.coords_pointed)

        if not isinstance(v, np.ndarray):
            v = np.array(v)

        assert v.ndim == len(self.shape_fixed), (v.ndim, len(self.shape_fixed))
        assert v.shape == self.shape_fixed, (v.shape, self.shape_fixed)

        v_nonzero = np.nonzero(v.flatten())[0]
        v_data = v.flatten()[v_nonzero]
        v_indices = v_nonzero

        zarr.save(self._get_key_path_data(k), v_data)
        zarr.save(self._get_key_path_indices(k), v_indices)

    def __getitem__(self, k):
        k = process_key(k, self.coords_pointed)

        path = self._get_key_path_data(k)
        if not path.exists():
            return np.zeros(self.shape_fixed)

        data = zarr.open(self._get_key_path_data(k), mode="r")
        indices = zarr.open(self._get_key_path_indices(k), mode="r")

        value = np.zeros(self.shape_fixed, dtype=data.dtype)
        if len(data) > 0:
            if value.ndim > 0:
                value[np.unravel_index(indices, value.shape)] = data
            else:
                return data[0]
        return value

    def sel(self, k=None):
        k = process_key(k, self.coords_pointed)
        ix = key_to_ix(k, self.coords_pointed, unwrap=True)

        v = []
        for ksubset in itertools.product(*[[i] if isinstance(i, (int, str, float)) else i for i in k]):
            v.append(self[ksubset])
        v = np.stack(v).reshape(tuple([len(i) for i in ix if not isinstance(i, int)] + list(self.shape_fixed)))
        return v

    def sel_xr(self, k=None):
        k = process_key(k, self.coords_pointed)
        v = self.sel(k)
        coords = key_to_coords(k, self.coords_pointed, self.coords_fixed)
        import xarray as xr

        return xr.DataArray(v, coords=coords)
