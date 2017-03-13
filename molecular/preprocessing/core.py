# -*- coding: utf-8 -*-

import dask.dataframe as dd
import logging as log
import multiprocessing as mp
import numpy as np
import pandas as pd

from ..util import mad

def discretise(X, nscale=1):
  ''' Discretise each feature distribution.
      X should be provided in the form [observations, features] '''

  _X = dd.from_pandas(X.T, npartitions=mp.cpu_count()+1)
  X = _X.apply(
    _discretise_series, axis=1, args=(nscale,), meta=X.T).compute().T

  return X

def _discretise_series(X, nscale=1):
  ''' Discretise pandas Series '''

  loc = X.median()
  scale = mad(X)
  # NOTE Adding float min to avoid same bin edge values in case of scale == 0
  precision = 3
  float_min = 1 / 10**precision
  bins = [
    -np.inf, loc - (scale * nscale) - float_min,
    loc + (scale * nscale) + float_min, np.inf]
  X = pd.cut(X, bins, labels=[-1, 0, 1], precision=precision)

  return X
