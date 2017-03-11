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

from molecular import discretise

log.basicConfig(filename='tests/test_core.log', level=log.DEBUG)

class TestCore(unittest.TestCase):

  def setUp(self):

    self.X = pd.DataFrame([
      [3,2,3],
      [2,3,1],
      [2,3,1]])

  def tearDown(self):
    pass

  def test_discretise(self):

    dX = pd.DataFrame([
      [1,-1,1],
      [0,0,0],
      [0,0,0]])

    _dX = discretise(self.X)
    assert dX.equals(_dX)
