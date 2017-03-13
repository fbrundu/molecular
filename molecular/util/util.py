# -*- coding: utf-8 -*-

import contextlib
import numpy as np
import os
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
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

def mc_roc_auc(average='macro'):
  ''' Multiclass ROC AUC '''

  def _mc_roc_auc(truth, pred, average="macro"):

    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return roc_auc_score(truth, pred, average=average)

  return make_scorer(_mclass_roc_auc)
