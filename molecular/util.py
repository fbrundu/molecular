# -*- coding: utf-8 -*-

import contextlib
import numpy as np
import os
import sys

@contextlib.contextmanager
def nostdout():
  savestdout = sys.stdout
  class Devnull(object):
    def write(self, _): pass
    def flush(self): pass
  sys.stdout = Devnull()
  try:
    yield
  finally:
    sys.stdout = savestdout

def mad(array):
  ''' Median Absolute Deviation: a "Robust" version of standard deviation.
      Indices variabililty of the sample.
      https://en.wikipedia.org/wiki/Median_absolute_deviation '''

  return np.median(np.abs(array - np.median(array)))
