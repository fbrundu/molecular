# -*- coding: utf-8 -*-

from imblearn.combine import SMOTETomek
import joblib
import keras as kk
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import logging as log
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (cross_val_predict, cross_val_score,
  GridSearchCV)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from molecular.plotting import Plot
from molecular.util import mad, mc_roc_auc


class _Model:

  def __init__(self, X, y):
    ''' Initialisation '''
    self.ntests = 1
    self.X = X
    self.y = y

    self._fix_imbalance()

    self.best_clf = None
    self.test_cm_norm = None

  def _fix_imbalance(self):
    ''' Fix imbalance of size between classes '''

    st = SMOTETomek()

    fX, fy = st.fit_sample(self.X.values, self.y.values.ravel())
    samples = [f'smp{i}' for i in range(fX.shape[0])]
    self.fX = pd.DataFrame(fX, index=samples, columns=self.X.columns)
    self.fy = pd.DataFrame(fy, index=samples, columns=self.y.columns)

    log.info(f'Keeping {fX.shape[0]} samples, {fX.shape[1]} features')

  @property
  def _param_grid(self):

    return {
      'dense_nodes': [
        np.linspace(self.fX.shape[1], np.unique(self.fy.iloc[:, 0]).size,
          nlayers, dtype=int)
        for nlayers in range(3, 3 + self.ntests)]}

  def _fit(self):

    nn = KerasClassifier(build_fn=self._arch, verbose=0, epochs=10)
    clf = GridSearchCV(
      nn, param_grid=self._param_grid, scoring='f1_macro', verbose=2, n_jobs=-1)
    clf.fit(self.fX.values, self.fy.values)
    self.best_clf = clf
    log.info(f'Best score: {clf.best_score_}')

  def _arch(self, dense_nodes):

    arch = Sequential()
    arch.add(BatchNormalization(input_shape=(self.fX.shape[1],)))
    for nodes in dense_nodes[:-1]:
      arch.add(Dense(nodes, activation='relu'))
      #arch.add(Dropout(0.5))
    arch.add(Dense(dense_nodes[-1], activation='sigmoid'))
    arch.compile(
      #optimizer=kk.optimizers.SGD(lr=0.01, decay=1e-3, clipvalue=1, momentum=1e-3),
      optimizer='sgd',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

    return arch

  def fit(self):

    self._fit()

  def _test(self, X_test, y_test):

    X_test = X_test.values.reshape(*X_test.shape)

    truth = y_test.values.ravel()
    pred = self.best_clf.best_estimator_.predict(X_test)
    proba = self.best_clf.best_estimator_.predict_proba(X_test)

    names = sorted(list(np.unique(truth)))
    self.test_cm = pd.DataFrame(confusion_matrix(truth, pred, labels=names),
      index=names, columns=names)
    self.test_cm_norm = self.test_cm.div(self.test_cm.sum(axis=1), axis=0)
    self.test_pred = pd.Series(pred, index=y_test.index)
    self.test_proba = pd.DataFrame(proba, index=y_test.index,
      columns=np.unique(y_test))

  def draw(self, what=[], **kwargs):

    if 'test' in what:
      fig = self._draw_test(**kwargs)

    return fig

  def _draw_test(self, X_test, y_test):

    if self.test_cm_norm is None:
      self._test(X_test, y_test)

    fig = Plot.heatmap(self.test_cm_norm, title='Normalised Confusion Matrix',
      xlabel='Predicted Label', ylabel='True Label')

    return fig


class CMSDeep17:

  def __init__(self, logfile=None):
    ''' Initialisation '''

    self.model = None
    self.X_test = None
    self.y_test = None

    np.random.seed(42)

    log.basicConfig(filename=logfile, level=log.INFO,
      format='%(asctime)s : %(levelname)8s : %(message)s (%(module)s.%(funcName)s)',
      datefmt='%Y-%m-%d %H:%M:%S')

  def deserialise(self, fpath):
    ''' Deserialise best model '''

    self.model = joblib.load(fpath)

  def serialise(self, fpath):
    ''' Serialise best model '''

    joblib.dump(self.model, fpath)

  def fit(self, X_path, y_path, X_test_path=None, y_test_path=None):

    self._load(X_path, y_path, X_test_path=X_test_path, y_test_path=y_test_path)
    self._grid_fit()

  def _load(self, X_path, y_path, X_test_path=None, y_test_path=None):
    ''' Data should be provided in log2 '''

    log.info(f'Loading data, X = {X_path}, y = {y_path}')

    self.X = pd.read_table(X_path, sep='\t', index_col=0)
    self.y = pd.read_table(y_path, sep='\t', index_col=0)

    if X_test_path is not None and y_test_path is not None:
      self.X_test = pd.read_table(X_test_path, sep='\t', index_col=0)
      self.y_test = pd.read_table(y_test_path, sep='\t', index_col=0)

      samples_test = self.X_test.index & self.y_test.index
      self.X_test = self.X_test.ix[samples_test]
      self.y_test = self.y_test.ix[samples_test]

    samples = self.X.index & self.y.index
    self.X = self.X.ix[samples]
    self.y = self.y.ix[samples]

    # NOTE same features in training and testing
    if self.X_test is not None:
      feats = self.X.columns & self.X_test.columns
      self.X = self.X[feats]
      self.X_test = self.X_test[feats]

    self._scale()

  def _scale(self):

    # NOTE single sample scaling ~ N(0, 1)
    sc = StandardScaler()
    sX = sc.fit_transform(self.X.T)
    self.X = pd.DataFrame(sX, index=self.X.columns, columns=self.X.index).T

    if self.X_test is not None:
      # NOTE single sample scaling ~ N(0, 1)
      sc = StandardScaler()
      sX = sc.fit_transform(self.X_test.T)
      self.X_test = pd.DataFrame(
        sX, index=self.X_test.columns, columns=self.X_test.index).T

  def _grid_fit(self):

    log.info('Starting Grid Search')
    self.model = _Model(self.X, self.y)
    self.model.fit()

  def draw_all(self, path):

    if self.X_test is not None and self.y_test is not None:
      fig = self.model.draw(what=['test'], X_test=self.X_test,
        y_test=self.y_test)
      fig.savefig(os.path.join(path, 'test_cm.pdf'), bbox_inches='tight')
