import numpy as np
import h5py

def sliceDataAlongAxis(data, fractions, axis):
    data_size = data.shape[0]
    fractions_ = np.zeros_like(fractions, dtype=int)

    total_size = 0
    for i, fraction in enumerate(fractions):
        total_size += int(data_size*fraction)
    remain = data_size-total_size

    slices = ()
    for i, fraction in enumerate(fractions):
        fractions_[i] = int(data_size*fraction)
        if i > 0:
            fractions_[i] += fractions_[i-1]
            slice = data.take(range(fractions_[i-1], fractions_[i]), axis)

        else:
            slice = data.take(range(0, fractions_[i]+remain), axis)

        slices += (slice,)

    return slices

class dataHandler:
    def __init__(self, file, variables):
        self.file = file
        self.variables = variables

        self.stack()

    def stack(self):
        with h5py.File(self.file, 'r') as f:
            self.h5 = f
            self.data = np.vstack([f[var][()] for var in self.variables])

            self.indexes = {}
            start_idx = 0
            for var in self.variables:
                end_idx = start_idx + f[var][()].shape[0]
                self.indexes[var] = np.arange(start_idx, end_idx)
                start_idx = end_idx

            if 'meshfile' in f.keys():
                self.meshfile = f['meshfile'][()]

    def get(self, variable):

        return self.data[self.indexes[variable], :]
    

import numpy as np
import xarray as xr

def split_dataset_named(
    ds: xr.Dataset,
    dim: str = "design_point",
    frac=(0.8, 0.10, 0.10),
    seed: int = 42,
    sort: bool = True
):
    """
    Divide um Dataset xarray ao longo de uma dimensão com coordenadas nomeadas
    em conjuntos de treino, validação e teste.

    Parâmetros:
        ds   : xarray.Dataset
            Dataset de entrada.
        dim  : str
            Nome da dimensão a ser dividida (ex: "design").
        frac : tuple of float
            Frações para treino, validação e teste. Deve somar aproximadamente 1.0.
        seed : int
            Semente para embaralhamento reprodutível.
        sort : bool
            Se True, ordena os conjuntos por nome.

    Retorna:
        ds_train, ds_val, ds_test : xarray.Dataset
    """
    
    coords = ds.coords[dim].values
    n_total = len(coords)
    idx = np.arange(n_total)
    np.random.seed(seed)
    np.random.shuffle(idx)

    n_train = int(frac[0] * n_total)
    n_val = int(frac[1] * n_total)

    train_labels = coords[idx[:n_train]]
    val_labels   = coords[idx[n_train:n_train + n_val]]
    test_labels  = coords[idx[n_train + n_val:]]

    
    if sort:
        import re

        def sort_by_number(labels):
            return sorted(labels, key=lambda x: int(re.search(r'\d+', x).group()))

        train_labels = sort_by_number(train_labels)
        val_labels   = sort_by_number(val_labels)
        test_labels  = sort_by_number(test_labels)

    return (
        ds.sel({dim: train_labels}),
        ds.sel({dim: val_labels}),
        ds.sel({dim: test_labels}),
    )

