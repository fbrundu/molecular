# -*- coding: utf-8 -*-

from imblearn.combine import SMOTETomek
import joblib
import logging as log
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
import statsmodels.sandbox.stats.multicomp as smc

from molecular.plotting import Plot
from molecular.preprocessing import FeatureConstruction, FeatureSelection
from molecular.util import mad, mc_roc_auc


class CMSForests16:

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
      self.cv_cm_norm = None
      self.clf_params = { 'random_state': 42, 'criterion': 'gini',
        'max_features': None }

    def _fix_imbalance(self):
      ''' Fix imbalance of size between classes '''

      st = SMOTETomek()

      fX, fy = st.fit_sample(self.X.values, self.y.values.ravel())
      samples = [f'smp{i}' for i in range(fX.shape[0])]
      self.fX = pd.DataFrame(fX, index=samples, columns=self.X.columns)
      self.fy = pd.DataFrame(fy, index=samples, columns=self.y.columns)

      log.debug(f'Keeping {fX.shape[0]} samples, {fX.shape[1]} features')

    def _fit(self):

      rfc = RandomForestClassifier(**self.clf_params)
      clf = GridSearchCV(
        rfc, self.params, scoring=mc_roc_auc(average='macro'), n_jobs=-1)
      clf.fit(self.fX.values, self.fy.values.ravel())

      index, columns = 'min_samples_split', 'n_estimators'
      scores = self._shape_scores(clf.grid_scores_, index, columns)
      self._select_best(scores, index, columns)

    def fit(self):

      self._fit()
      self._cv()

    def _cv(self):

      self.best_clf = RandomForestClassifier(**self.clf_params)
      self._cv_score()
      self._cv_pred()

    def _cv_score(self):

      scores = cross_val_score(self.best_clf, self.fX.values,
        self.fy.values.ravel(), scoring=mc_roc_auc(average='macro'))
      self.best_score_med = round(np.median(scores), 3)
      self.best_score_mad = round(mad(scores), 3)

    def _cv_pred(self):

      truth = self.fy.values.ravel()
      pred = cross_val_predict(estimator=self.best_clf, X=self.fX.values,
        y=self.fy.values.ravel())
      proba = cross_val_predict(estimator=self.best_clf, X=self.fX.values,
        y=self.fy.values.ravel(), method='predict_proba')
      names = sorted(list(np.unique(truth)))

      self.cv_cm = pd.DataFrame(confusion_matrix(truth, pred, labels=names),
        index=names, columns=names)
      self.cv_cm_norm = self.cv_cm.div(self.cv_cm.sum(axis=1), axis=0)
      self.cv_truth = pd.Series(truth, index=self.fy.index)
      self.cv_pred = pd.Series(pred, index=self.fy.index)
      self.cv_proba = pd.DataFrame(proba, index=self.fy.index,
        columns=np.unique(self.fy))

    def _test(self, X_test, y_test):

      self.best_clf.fit(X=self.fX.values, y=self.fy.values)

      truth = y_test.values.ravel()
      pred = self.best_clf.predict()
      proba = self.best_clf.predict_proba()

      names = sorted(list(np.unique(truth)))
      self.test_cm = pd.DataFrame(confusion_matrix(truth, pred, labels=names),
        index=names, columns=names)
      self.test_cm_norm = self.test_cm.div(self.test_cm.sum(axis=1), axis=0)
      self.test_pred = pd.Series(pred, index=y_test.index)
      self.test_proba = pd.DataFrame(proba, index=y_test.index,
        columns=np.unique(y_test))

    def _shape_scores(self, scores, index, columns):

      scores = np.array([x[1] for x in scores])
      scores = scores.reshape(len(self.params[index]),
        len(self.params[columns]))
      return pd.DataFrame(scores, index=self.params[index],
        columns=self.params[columns])

    def _select_best(self, scores, index, columns):

      df = pd.concat([scores.median().round(3), scores.apply(mad).round(3)],
        axis=1)
      df = df.reset_index()
      df.columns = [columns, 'Median', 'MAD']
      df = df.sort_values(by=['Median', 'MAD', columns],
        ascending=[False, True, True])
      self.best_ntrees = int(df.iloc[0][columns])

      df = pd.concat([scores.T.median().round(3), scores.T.apply(mad).round(3)],
        axis=1)
      df = df.reset_index()
      df.columns = [index, 'Median', 'MAD']
      df = df.sort_values(by=['Median', 'MAD', index],
        ascending=[False, True, False])
      self.best_minsmp = int(df.iloc[0][index])

      self.clf_params = {**self.clf_params, **{
        'n_estimators': self.best_ntrees,
        'min_samples_split': self.best_minsmp,
      }}

    @property
    def importance(self):

      if self.best_clf is None:
        raise Exception('Best classifier not fitted')

      if self._importance is None:
        log.debug('Computing importance')

        self.best_clf.fit(self.fX.values, self.fy.values.ravel())
        self._importance = self.best_clf.feature_importances_
        self._importance = pd.DataFrame(self._importance, index=self.fX.columns,
          columns=['Importance'])
        _forest = self.best_clf.estimators_
        self._importance['StD'] = np.std([
          tree.feature_importances_ for tree in _forest], axis=0)
        self._importance = self._importance.sort_values(by=['Importance'],
          ascending=False)

        log.debug('Computing importance significance')

      return self._importance

    def draw(self, what=[], **kwargs):

      if 'importance' in what:
        fig = self._draw_importance(**kwargs)
      elif 'distribution' in what:
        fig = self._draw_distribution(**kwargs)
      elif 'confusion' in what:
        fig = self._draw_confusion(**kwargs)
      elif 'roc' in what:
        fig = self._draw_roc(**kwargs)
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

    def _draw_confusion(self):

      if self.cv_cm_norm is None:
        self._cv_pred()

      fig = Plot.heatmap(self.cv_cm_norm, title='Normalised Confusion Matrix',
        xlabel='Predicted Label', ylabel='True Label')

      return fig

    def _draw_test(self, X_test, y_test):

      if self.test_cm_norm is None:
        self._test(X_test, y_test)

      fig = Plot.heatmap(self.test_cm_norm, title='Normalised Confusion Matrix',
        xlabel='Predicted Label', ylabel='True Label')

      return fig

    def _draw_roc(self):

      if self.cv_cm_norm is None:
        self._cv_pred()

      fig = Plot.roc(self.fy, self.cv_proba)

      return fig

    @property
    def params(self):

      return {
        'n_estimators': np.linspace(100, 550, self.ntests, dtype=int),
        'min_samples_split': np.linspace(2, self._mincls, self.ntests,
          dtype=int) }

  def __init__(self, discretise=True, nscale=1):
    ''' Initialisation '''

    self._max_feat = None
    self._ntests = 10
    self.model = None
    self.n = None
    self.discretise = discretise
    self.nscale = nscale
    self.fcons_top = 10000
    self.X_test = None
    self.y_test = None

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
      self._clean(clean=('perc_', 'l2r_', 'diff_'))
      self._build(build_subset=build_subset, build_method=build_method)

    if include:
      keep = include
      clean = ('')

    self._clean(exclude=exclude, keep=keep, clean=clean)
    self._grid_fit()

  def _load(self, X_path, y_path, subsample=[], X_test_path=None,
      y_test_path=None):
    ''' Data should be provided in log2 '''

    log.debug(f'Loading data, X = {X_path}, y = {y_path}')

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

    log.debug('Feature selection')
    fs = FeatureSelection(X=self.X, y=self.y)
    n = min(self.max_feat, self.X.shape[1])
    fs.fit(selector=['mRMR'], n=n, discretise=self.discretise,
      nscale=self.nscale)
    sX = fs.X

    all_feat = False

    for n in self.nfeats:
      if not all_feat:
        log.debug(f'Fitting model with n features = {n}')

        log.debug('Starting Grid Search')
        gcv = _Model(sX.iloc[:, :n], self.y)
        gcv.fit()

        self._update_model(gcv, n)

        if self.X.shape[1] <= n:
          all_feat = True

  def draw_all(self, path):

    fig = self.model.draw(what=['importance'])
    fig.savefig(os.path.join(path, 'importance.pdf'), bbox_inches='tight')

    for feat in self.model.importance.index[:6]:
      fig = self.model.draw(what=['distribution'], feat=feat,
        xlabel=self.y.columns[0])
      fig.savefig(os.path.join(path, feat + '.pdf'), bbox_inches='tight')

    fig = self.model.draw(what=['confusion'])
    fig.savefig(os.path.join(path, 'cm.pdf'), bbox_inches='tight')

    fig = self.model.draw(what=['roc'])
    fig.savefig(os.path.join(path, 'roc.pdf'), bbox_inches='tight')

    if self.X_test is not None and self.y_test is not None:
      fig = self.model.draw(what=['test'], X_test=self.X_test,
        y_test=self.y_test)
      fig.savefig(os.path.join(path, 'test_cm.pdf'), bbox_inches='tight')

  @property
  def nfeats(self):

    ns = np.linspace(max(self.max_feat // self._ntests, 3), self.max_feat,
      self._ntests, dtype=int)

    return np.unique(ns)

  def _update_model(self, gcv, n):

    def _better_score(nmed, omed, nmad, omad, nn, on):
      if ((nmed > omed) or (nmed == omed and nmad < omad) or
          (nmed == omed and nmad == omad and nn > on)):
        return True
      else:
        return False

    if self.model is None:
      log.debug('Inserting first model')
      log.debug(f'Score median is {gcv.best_score_med}')
      log.debug(f'Score MAD is {gcv.best_score_mad}')
      self.model = gcv
      self.n = n
    elif _better_score(gcv.best_score_med, self.model.best_score_med,
        gcv.best_score_mad, self.model.best_score_mad, n, self.n):
      log.debug('Updating with better model')
      log.debug(f'Score median is {gcv.best_score_med}')
      log.debug(f'Score MAD is {gcv.best_score_mad}')
      self.model = gcv
      self.n = n

  def _clean(self, clean=tuple(), keep=tuple(), exclude=tuple()):

    cols = [c for c in self.X.columns
      if c.startswith(keep) or not c.startswith(clean)]
    self.X = self.X[cols]

    if exclude is not None:
      cols = self.X.columns
      for e in exclude:
        cols = [c for c in cols if e not in c]
      self.X = self.X[cols]

    log.debug(
      f'Keeping {self.X.shape[0]} samples and {self.X.shape[1]} features')

  def _build(self, build_subset=tuple(), build_method='ratio'):

    if build_subset:
      build_subset = set(build_subset) & set(self.X.columns)
      self.X = self.X[list(build_subset)]
      log.debug(f'Building from {self.X.shape[1]} features')

    log.debug('Building max {self.fcons_top} final features')
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
