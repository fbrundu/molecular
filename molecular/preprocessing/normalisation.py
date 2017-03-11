# -*- coding: utf-8 -*-

import dask.dataframe as dd
import logging as log
import multiprocessing as mp
import pandas as pd
import scipy.stats as ss


# TODO testing
class Normalisation:

  def __init__(self, X):
    ''' Initialisation. X should be provided in the form
        [observations, features] '''
    self.X = X.copy()
    self.X_means = None

  def quantile(self, X_means=None):
    ''' Quantile normalises each observation distribution '''

    def _rank_prefix(x):

      x = x.rank(method="min").astype(int).astype(str)
      x = x.apply(lambda y: "r" + y)

      return x

    if X_means is not None:
      self.X_means = X_means
    else:
      sX = dd.from_pandas(self.X, npartitions=mp.cpu_count()+1)
      sX = sX.apply(sorted, axis=1, meta=self.X).compute()
      self.X_means = sX.mean().tolist()

    rX = dd.from_pandas(self.X, npartitions=mp.cpu_count()+1)
    rX = rX.apply(_rank_prefix, axis=1, meta=X)
    self.X = rX.compute()

    for i in range(len(X_means)):
      self.X = self.X.replace(
        to_replace="r" + str(i + 1), value=self.X_means[i])

  def standardise(self):
    ''' Standardises each observation distribution '''

    dX = dd.from_pandas(self.X, npartitions=mp.cpu_count()+1)
    self.X = dX.apply(ss.zscore, axis=1, meta=self.X).compute()

  def fit(self, normaliser, **kwargs):
    ''' Normalise data '''

    if normaliser == 'quantile':
      self._quantile(**kwargs)
    elif normaliser == 'zscore':
      self._zscore(**kwargs)
    else:
      raise ValueError('Normaliser {normaliser} not implemented')

    return self
