# -*- coding: utf-8 -*-

import logging as log
import pymrmr

from .core import discretise

class FeatureSelection:

  def __init__(self, X, y):
    ''' Initialisation '''
    self.X = X.copy()
    self.y = y.copy()

  def _mRMR(self, n, method='MIQ', is_discrete=True, nscale=1):
    ''' minimum Redundancy Maximum Relevance algorithm '''

    sX = self.X.copy()

    if not is_discrete:
      log.info(f'Discretising X using scale = scale * {nscale}')
      sX = discretise(sX, nscale)

    sX.insert(0, self.y.columns[0], self.y.iloc[:, 0])

    log.info(f'Starting mRMR ({method}, n={n})')
    feats = pymrmr.mRMR(sX, 'MIQ', n)

    log.info(f'Updating dataset, {len(feats)} features')
    self.X = self.X[feats]

  def fit(self, selector, **kwargs):
    ''' Select features '''

    if selector == 'mRMR':
      self._mRMR(**kwargs)
    else:
      raise ValueError('Selector {selector} not implemented')

    return self
