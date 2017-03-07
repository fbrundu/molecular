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
    self.top = top
    self.chunksize = chunksize

    self._ignore = ('l2r',)
    self._njobs = mp.cpu_count() + 1

  def _ex_args(self, i, ff):
    ''' Produces data chunk for each executor '''
    args = []

    for j in range(i, i + self._njobs):
      try:
        args += [self.X.loc[:, ff[j]:].copy()]
      except:
        log.warning(f'No map args for j {j}')

    return args

  def _first_feat(self):
    ''' List of first feature for each chunk '''

    feat = [c for c in self.X.columns if not c.startswith(self._ignore)]
    return [feat[i] for i in range(0, len(feat) - 1, self.chunksize)]

  def _ratio(self, **kwargs):
    ''' Computes log2-ratio between pair of features '''
    ff = self._first_feat()
    nX = None

    with cf.ProcessPoolExecutor(max_workers=self._njobs) as executor:
      log.info(f'Starting Pool: {self._njobs} workers, {len(ff)} chunks')

      for i in range(0, len(ff), self._njobs):
        log.info(f'Chunk {i} of {len(ff)}')

        cX = [
          f for f in executor.map(self._ratio_internal, self._ex_args(i, ff))]

        nX = self._merge_and_select(nX, cX)

    self.X = pd.concat([self.X, nX], axis=1)

  def _merge_and_select(self, nX, cX):
    ''' Merges the results of the new chunk with previous results, and selects
        top features '''

    nX = pd.concat([nX] + cX, axis=1)
    nX = FeatureConstruction._select(nX, self.top, y=self.y)

    return nX

  @staticmethod
  def _select(X, top, y):
    ''' Selects top features based on Kruskal-Wallis test p-value '''

    X = X.copy()
    feats = pd.DataFrame(index=X.columns, columns=['Value'])
    X.loc[:, 'Class'] = y.ix[X.index].astype(int)
    ascending = True

    for feat in feats.index:
      nX = X[[feat, 'Class']].copy()
      groups = [v[v.columns[0]].values for k, v in nX.groupby(['Class'])]
      log.debug('\n' + str(groups))
      _, p_val = ss.kruskal(*groups)
      feats.loc[feat, 'Value'] = p_val

    log.debug('\n' + str(feats['Value']))

    feats = feats.sort_values(by=['Value'], ascending=[ascending])

    return X[feats.head(n=top).index]

  def _ratio_internal(self, X):
    ''' Log2-ratio routine executed by each worker '''

    feats = X.columns

    for i in range(self.chunksize):
      for j in range(i + 1, len(feats)):
        X[f'l2r_{feats[i]}_{feats[j]}'] = X[feats[i]] - X[feats[j]]

    X = X[[f for f in X.columns if f.startswith('l2r_')]]

    if self.top is not None:
      X = FeatureConstruction._select(X, self.top, y=self.y)

    return X

  def _ratio_spec(self, ratios):
    ''' Build specific ratios '''

    for r in ratios:
      feats = r.partition('_')[2]
      feats = feats.split('_')
      self.X[f'l2r_{feats[0]}_{feats[1]}'] = (
        self.X[feats[0]] - self.X[feats[1]])

  def fit(self, constructor, **kwargs):
    ''' Build features '''

    if constructor == 'ratio':
      self._ratio(**kwargs)
    elif constructor == 'ratio_spec':
      self._ratio_spec(**kwargs)
    else:
      raise ValueError(f'Constructor {constructor} not implemented')

    return self
