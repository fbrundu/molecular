# -*- coding: utf-8 -*-

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import joblib
import logging as log
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer

from molecular.plotting import Plot
from molecular.preprocessing import FeatureConstruction, FeatureSelection

# FIXME deprecated
# fix when https://github.com/scikit-learn/scikit-learn/pull/7663 is ready
class Score:

  @staticmethod
  def _mclass_roc_auc(truth, pred, average="macro"):

    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return roc_auc_score(truth, pred, average=average)

  def get(scorer, average="macro"):

    if scorer == "roc_auc":
      return make_scorer(Score._mclass_roc_auc)
####


class _Model:

  def __init__(self, X, y):
    ''' Initialisation '''

    self._ntrees_l = []
    self._minsmp_l = []
    self.ntests = 15
    self.X = X
    self.y = y

    self._fix_imbalance()

    self._mincls = y.iloc[:, 0].value_counts().min()
    self.best_clf = None
    self._importance = None
    self._ntop = 40
    self.test_cm_norm = None
    self.clf_params = { 'random_state': 42, 'criterion': 'gini',
      'max_features': None }

  def _fix_imbalance(self):
    ''' Fix imbalance of size between classes '''

    # FIXME find best ratio
    card = self.y.iloc[:,0].value_counts()
    ratio = card.max() / card.min()
    
    if ratio < 1.5:
      st = SMOTETomek()
    else: 
      st = RandomUnderSampler()
    ###

    fX, fy = st.fit_sample(self.X.values, self.y.values.ravel())
    samples = [f'smp{i}' for i in range(fX.shape[0])]
    self.fX = pd.DataFrame(fX, index=samples, columns=self.X.columns)
    self.fy = pd.DataFrame(fy, index=samples, columns=self.y.columns)

    log.info(f'Keeping {fX.shape[0]} samples, {fX.shape[1]} features')

  def _fit(self):

    rfc = RandomForestClassifier(**self.clf_params)
    clf = GridSearchCV(rfc, param_grid=self._param_grid,
      scoring=Score.get("roc_auc", average="macro"), verbose=2, n_jobs=-1)
    clf.fit(self.fX.values, self.fy.values.ravel())
    self.best_clf = clf
    log.info(f'Best score: {clf.best_score_}')

  def fit(self):

    self._fit()

  def _test(self, X_test, y_test):

    truth = y_test.values.ravel()
    pred = self.best_clf.best_estimator_.predict(X_test[self.fX.columns])
    proba = self.best_clf.best_estimator_.predict_proba(X_test[self.fX.columns])

    names = sorted(list(np.unique(truth)))
    self.test_cm = pd.DataFrame(confusion_matrix(truth, pred, labels=names),
      index=names, columns=names)
    self.test_cm_norm = self.test_cm.div(self.test_cm.sum(axis=1), axis=0)
    self.test_pred = pd.Series(pred, index=y_test.index)
    self.test_proba = pd.DataFrame(proba, index=y_test.index,
      columns=np.unique(y_test))

  @property
  def importance(self):

    if self.best_clf is None:
      raise Exception('Best classifier not fitted')

    if self._importance is None:
      log.info('Computing importance')

      self._importance = self.best_clf.best_estimator_.feature_importances_
      self._importance = pd.DataFrame(self._importance, index=self.fX.columns,
        columns=['Importance'])
      _forest = self.best_clf.best_estimator_.estimators_
      self._importance['StD'] = np.std([
        tree.feature_importances_ for tree in _forest], axis=0)
      self._importance = self._importance.sort_values(by=['Importance'],
        ascending=False)

    return self._importance

  def draw(self, what=[], **kwargs):

    fig = None
    if 'importance' in what:
      fig = self._draw_importance(**kwargs)
    elif 'distribution' in what:
      fig = self._draw_distribution(**kwargs)
    elif 'test' in what:
      fig = self._draw_test(**kwargs)

    return fig

  def _draw_importance(self):

    data = self.importance.ix[:self._ntop]

    fig = Plot.bar(data, title='Features importance')

    return fig

  def _draw_distribution(self, feat, xlabel=''):

    x = self.y.ix[:, 0]
    x.name = xlabel
    y = self.X.ix[:, feat]
    y.name = feat

    data = pd.concat([x, y], axis=1)

    ylim_max = self.X[feat].abs().max() + .1
    ylim = (-ylim_max, ylim_max)

    fig = Plot.box(data, x=x.name, y=y.name, ylim=ylim)

    return fig

  def _draw_test(self, X_test, y_test):

    if self.test_cm_norm is None:
      self._test(X_test, y_test)

    fig = Plot.heatmap(self.test_cm_norm, title='Normalised Confusion Matrix',
      xlabel='Predicted Label', ylabel='True Label')

    return fig

  @property
  def _param_grid(self):

    return {
      'n_estimators': np.linspace(100, 550, self.ntests, dtype=int),
      'min_samples_split': np.linspace(2, self._mincls, self.ntests,
        dtype=int) }


class CMSForests16:

  def __init__(self, is_discrete=False, nscale=1, logfile=None):
    ''' Initialisation '''

    self._max_feat = None
    self._ntests = 10
    self.model = None
    self.n = None
    self.is_discrete = is_discrete
    self.nscale = nscale
    self.fcons_top = 10000
    self.X_test = None
    self.y_test = None

    log.basicConfig(filename=logfile, level=log.INFO,
      format='%(asctime)s : %(levelname)8s : %(message)s (%(module)s.%(funcName)s)',
      datefmt='%Y-%m-%d %H:%M:%S')

  def deserialise(self, fpath):
    ''' Deserialise best model '''

    self.model = joblib.load(fpath)

  def serialise(self, fpath):
    ''' Serialise best model '''

    joblib.dump(self.model, fpath)

  def fit(self, X_path, y_path, build=True, exclude=tuple(), subsample=[],
      keep=('l2r_',), include=tuple(), build_subset=tuple(), clean=tuple(),
      build_method='ratio', X_test_path=None, y_test_path=None):

    self._load(X_path, y_path, subsample=subsample, X_test_path=X_test_path,
      y_test_path=y_test_path)

    if build:
      CMSForests16._clean(self.X, clean=('perc_', 'l2r_', 'diff_'))
      self._build(build_subset=build_subset, build_method=build_method)

    if include:
      keep = include
      clean = ('')

    self.X = CMSForests16._clean(self.X, exclude=exclude, keep=keep,
      clean=clean)
    self._grid_fit()
    self._prepare_test()
    self.X_test = CMSForests16._clean(self.X_test, exclude=exclude,
      keep=keep, clean=clean)

  def _prepare_test(self):

    log.info(f'Building {self.max_feat} features for test dataset')
    fc = FeatureConstruction(X=self.X_test, top=self.fcons_top, y=self.y_test)
    fc.fit(constructor='ratio_spec', ratios=list(self.model.fX.columns))
    self.X_test = fc.X

  def _load(self, X_path, y_path, subsample=[], X_test_path=None,
      y_test_path=None):
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

    if len(subsample) > 0:
      subsample = set(self.X.index) & set(subsample)
      subsample = list(subsample)
      self.X = self.X.ix[subsample]
      self.y = self.y.ix[subsample]

  def _grid_fit(self):

    log.info('Feature selection')
    fs = FeatureSelection(X=self.X, y=self.y)
    fs.fit(selector='mRMR', n=self.max_feat, is_discrete=self.is_discrete,
      nscale=self.nscale)
    sX = fs.X

    log.info(f'Fitting model with n features = {self.max_feat}')

    log.info('Starting Grid Search')
    self.model = _Model(sX, self.y)
    self.model.fit()

  def draw_all(self, path):

    fig = self.model.draw(what=['importance'])
    fig.savefig(os.path.join(path, 'importance.pdf'), bbox_inches='tight')

    for feat in self.model.importance.index[:6]:
      fig = self.model.draw(what=['distribution'], feat=feat,
        xlabel=self.y.columns[0])
      fig.savefig(os.path.join(path, feat + '.pdf'), bbox_inches='tight')

    if self.X_test is not None and self.y_test is not None:
      fig = self.model.draw(what=['test'], X_test=self.X_test,
        y_test=self.y_test)
      fig.savefig(os.path.join(path, 'test_cm.pdf'), bbox_inches='tight')

  @staticmethod
  def _clean(X, clean=tuple(), keep=tuple(), exclude=tuple()):

    cols = [c for c in X.columns
      if c.startswith(keep) or not c.startswith(clean)]
    X = X[cols]

    import ipdb; ipdb.set_trace()
    if exclude is not None:
      cols = X.columns
      for e in exclude:
        cols = [c for c in cols if e not in c]
      X = X[cols]

    log.info(f'Keeping {X.shape[0]} samples and {X.shape[1]} features')

    return X

  def _build(self, build_subset=tuple(), build_method='ratio'):

    if build_subset:
      build_subset = set(build_subset) & set(self.X.columns)
      self.X = self.X[list(build_subset)]
      log.info(f'Building from {self.X.shape[1]} features')

    log.info(f'Building max {self.fcons_top} final features')
    fc = FeatureConstruction(X=self.X, top=self.fcons_top, y=self.y)
    fc.fit(constructor='ratio')
    self.X = fc.X

  @property
  def max_feat(self):
    ''' Define maximum number of features to select. However, no more than
        50 features are selected.
        From: I. M. Johnstone and D. M. Titterington, “Statistical
          challenges of high-dimensional data.,” Philos Trans A Math Phys Eng
          Sci, vol. 367, no. 1906, pp. 4237–4253, Nov. 2009. '''

    if self._max_feat is None:
      card = self.y.iloc[:, 0].value_counts()
      self._max_feat = int((card.min() * card.count()) / 5)

    # no more than 50 features
    return min(self.X.shape[1], self._max_feat, 50)
