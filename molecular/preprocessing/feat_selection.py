# -*- coding: utf-8 -*-

import logging as log
import pandas as pd
import pymrmr

from .core import discretise

class FeatureSelection:

  def __init__(self, X, y):
    ''' Initialisation '''
    self.X = X.copy()
    self.y = y.copy()

  def _mRMR(self, n, method='MIQ', discretise=False, nscale=1):
    ''' minimum Redundancy Maximum Relevance algorithm '''

    _X = self.X.copy()

    if discretise:
      log.debug(f'Discretising X using scale = scale * {nscale}')
      _X = discretise(_X, nscale)

    _X.insert(0, self.y.columns[0], self.y.iloc[:, 0])

    log.debug(f'Starting mRMR ({method}, n={n})')
    _feats = pymrmr.mRMR(_X, 'MIQ', n)

    log.debug(f'Updating dataset, {len(_feats)} features')
    self.X = self.X[_feats]

  def fit(self, selector, **kwargs):
    ''' Select features '''

    if selector == 'mRMR':
      self._mRMR(**kwargs)
    else:
      raise ValueError('Selector {selector} not implemented')

    return self
