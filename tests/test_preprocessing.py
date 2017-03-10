#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_preprocessing
----------------------------------

Tests for `preprocessing` submodule.
"""


import logging as log
import pandas as pd
import sys
import unittest

from molecular import FeatureConstruction

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
