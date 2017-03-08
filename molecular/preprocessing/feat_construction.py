# -*- coding: utf-8 -*-

import concurrent.futures as cf
import logging as log
import multiprocessing as mp
import pandas as pd
import scipy.stats as ss


class FeatureConstruction:

  def __init__(self, X, y, top, chunksize=10):
    ''' Initialisation '''

    self.X = X.copy()
    self.y = y.iloc[:, 0].copy()
    self._ignore = ('l2r', )
    self._top = top
    self._chunksize = chunksize
    self._njobs = mp.cpu_count() + 1

    self._ff = None
    self._nX = None

  def _ex_args(self, i):
    ''' Produces data chunk for each executor '''
    args = []

    for j in range(i, i + self._njobs):
      try:
        args += [self.X.loc[:, self._first_feat[j]:].copy()]
      except:
        log.error(f'No map args for j {j}')

    return args

  @property
  def _first_feat(self):
    ''' List of first feature for each chunk '''

    if self._ff is None:
      feat = [c for c in self.X.columns if not c.startswith(self._ignore)]
      self._ff = [feat[i] for i in range(0, len(feat) - 1, self._chunksize)]

    return self._ff

  def _ratio(self, **kwargs):
    ''' Computes log2-ratio between pair of features '''

    with cf.ProcessPoolExecutor(max_workers=self._njobs) as executor:
      log.debug(
        f'Starting Pool: {self._njobs} workers, {len(self._first_feat)} chunks')

      for i in range(0, len(self._first_feat), self._njobs):
        log.debug(f'Chunk {i} of {len(self._first_feat)}')

        f_X = [f for f in executor.map(self._ratio_internal, self._ex_args(i))]

        self._merge_and_select(f_X)

    self.X = pd.concat([self.X, self._nX], axis=1)

  def _merge_and_select(self, f_X):
    ''' Merges the results of the new chunk with previous results, and selects
        top features '''

    if self._nX is None:
      self._nX = pd.concat(f_X, axis=1)
    else:
      self._nX = pd.concat([self._nX] + f_X, axis=1)

    if self._top is not None:
      self._nX = FeatureConstruction._select(self._nX, self._top, y=self.y)

  @staticmethod
  def _select(X, top, y=None):
    ''' Selects top features based on Kruskal-Wallis test p-value '''

    X = X.copy()
    feats = pd.DataFrame(index=X.columns, columns=['Value'])
    X.loc[:, 'Class'] = y.ix[X.index].astype(int)
    ascending = True

    for feat in feats.index:
      nX = X[[feat, 'Class']].copy()
      groups = [v[v.columns[0]].values for k, v in nX.groupby(['Class'])]
      _, p_val = ss.kruskal(*groups)
      feats.loc[feat, "Value"] = p_val

    feats = feats.sort_values(by=['Value'], ascending=[ascending])

    return X[feats.head(n=top).index]

  def _ratio_internal(self, X):
    ''' Log2-ratio routine executed by each worker '''

    feats = X.columns

    for i in range(self._chunksize):
      for j in range(i + 1, len(feats)):
        X[f'l2r_{feats[i]}_{feats[j]}'] = X[feats[i]] - X[feats[j]]

    X = X[[f for f in X.columns if f.startswith('l2r_')]]

    if self._top is not None:
      X = FeatureConstruction._select(X, self._top, y=self.y)

    return X

  def build(self, methods=[], **kwargs):
    ''' Entry point for feature construction execution '''

    for m in methods:
      if m == 'ratio':
        self._ratio(**kwargs)
      else:
        log.info(f'Method {m} not recognised')

    return self.X
