#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_preprocessing
----------------------------------

Tests for `preprocessing` submodule.
"""


import io
import logging as log
import pandas as pd
import requests
import sys
import unittest


from molecular.preprocessing import (FeatureConstruction, FeatureSelection)

log.basicConfig(filename='tests/test_preprocessing.log', level=log.DEBUG)

class TestFeatureConstruction(unittest.TestCase):

  def setUp(self):

    self.X = pd.DataFrame(
      [[3,2,3,1,2,3,4,5,6],
       [2,3,1,2,3,1,5,6,4],
       [2,3,1,2,3,1,5,6,4],
       [2,3,1,2,3,1,5,6,4],
       [2,3,5,2,2,1,3,6,4],
       [2,3,1,2,3,2,5,6,4],
       [2,3,1,2,3,1,5,6,4],
       [2,3,1,6,3,1,3,6,4],
       [2,3,1,4,2,3,2,1,4]],
      columns=[f'gene{i}' for i in range(9)])
    self.y = pd.DataFrame({'Class': [1,2,1,1,2,1,2,1,2]})

    self.top = 10
    chunksize = 2

    self.fc = FeatureConstruction(
      X=self.X, y=self.y, top=self.top, chunksize=chunksize)

    self.fc = self.fc.fit(constructor='ratio')
    self.nX = self.fc.X

  def tearDown(self):
    pass

  def test_ratio_dimensions(self):

    n_newfeats = min(self.top, sum(range(self.X.shape[1])))
    assert self.nX.shape[1] == (self.X.shape[1] + n_newfeats)

  def test_ratio_values(self):

    for feat in [f for f in self.nX.columns if f.startswith('l2r_')]:
      feat_s = feat.split('_')
      assert ((self.nX[feat_s[1]] - self.nX[feat_s[2]]) == self.nX[feat]).all()

  def test_ratio_dimensions_x2(self):

    self.fc.X = self.X
    self.fc = self.fc.fit(constructor='ratio')
    self.nX = self.fc.X

    n_newfeats = min(self.top, sum(range(self.X.shape[1])))
    assert self.nX.shape[1] == (self.X.shape[1] + n_newfeats)

  def test_ratio_values_x2(self):

    self.fc.X = self.X
    self.fc = self.fc.fit(constructor='ratio')
    self.nX = self.fc.X

    for feat in [f for f in self.nX.columns if f.startswith('l2r_')]:
      feat_s = feat.split('_')
      assert ((self.nX[feat_s[1]] - self.nX[feat_s[2]]) == self.nX[feat]).all()


class TestFeatureSelection(unittest.TestCase):

  def setUp(self):

    data_url = 'http://home.penglab.com/proj/mRMR/test_colon_s3.csv'
    data = requests.get(data_url).content
    data = pd.read_csv(io.StringIO(data.decode('utf-8')))

    self.feats = {
      10: set(['v765', 'v1123', 'v1772', 'v286', 'v467', 'v377', 'v513',
        'v1325', 'v1972', 'v1412']),
      20: set(['v765', 'v1123', 'v1772', 'v286', 'v467', 'v377', 'v513',
        'v1325', 'v1972', 'v1412', 'v1381', 'v897', 'v1671', 'v1582', 'v1423',
        'v317', 'v249', 'v1473', 'v1346', 'v125'])}

    self.X = data.loc[:, data.columns[1:]]
    self.y = data.iloc[:, [0]]

  def tearDown(self):
    pass

  def test_mRMR(self):

    for n in self.feats.keys():
      self.fs = FeatureSelection(X=self.X, y=self.y)
      self.fs = self.fs.fit(selector='mRMR', n=n, method='MIQ',
        discretise=False)
      log.debug(self.fs.X.columns)
      log.debug(self.feats[n])
      assert set(self.fs.X.columns) == set(self.feats[n])
