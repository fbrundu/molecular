# -*- coding: utf-8 -*-

import contextlib
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
