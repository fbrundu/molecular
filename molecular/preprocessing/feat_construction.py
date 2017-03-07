# -*- coding: utf-8 -*-

import concurrent.futures as cf
import multiprocessing as mp
import logging

# TODO logging

class FeatureConstruction:

    def __init__(self, X, y, log2_values, top):

        self.X = X.copy()
        self.y = y.iloc[:, 0].copy()
        self.ignore = ('l2r', )
        self.log2_values = log2_values
        self.top = top
        self.chunksize = 10
        self.order_by = None

    def _ratio(self, order_by='kruskal', **kwargs):

        self.order_by = order_by

        features = [c for c in self.X.columns if not c.startswith(self.ignore)]

        njobs = mp.cpu_count() + 1
        with cf.ProcessPoolExecutor(max_workers=njobs) as executor:
            log.debug(f'Starting ProcessPool: {njobs} workers')
            first_feat = [
              features[i] for i in range(0, len(features)-1, self.chunksize)]
            log.debug(f'There are {len(first_feat)} chunks')

            new_X = None
            for i in range(0, len(first_feat), njobs):
                log.debug(f'Chunk {i} of {len(first_feat}')

                map_args = []
                for j in range(i, i + njobs):
                    try:
                        map_args += [self.X.loc[:, first_feat[j]:].copy()]
                    except:
                        log.error('No map args for j {j}')

                f_X = [f for f in executor.map(self._ratio_internal, map_args)]

                if new_X is None:
                    new_X = pd.concat(f_X, axis=1)
                else:
                    new_X = pd.concat([new_X] + f_X, axis=1)

                if self.top is not None:
                    new_X = FeatureConstruction._top(
                      new_X, self.top, y=self.y, order_by=self.order_by)

            self.X = pd.concat([self.X, new_X], axis=1)

    @staticmethod
    def _top(X, top, order_by='kruskal', y=None):

        X = X.copy()

        feats = pd.DataFrame(index=X.columns, columns=['Value'])

        X.loc[:, 'Class'] = y.ix[X.index].astype(int)
        if order_by == 'kruskal':
            ascending = True

        for feat_name in feats.index:
            X_local = X[[feat_name, 'Class']].copy()
            groups = [
              v[v.columns[0]].values for k, v in X_local.groupby(['Class'])]
            _, p_val = ss.kruskal(*groups)
            if order_by == 'kruskal':
                feats.loc[feat_name, "Value"] = p_val

        feats = feats.sort_values(by=['Value'], ascending=[ascending])
        feats_l = feats.head(n=top).index
        X = X[feats_l]

        return X

    def _ratio_internal(self, X):

        features = X.columns

        for i, feat_i in enumerate(features[:self.chunksize]):
            for j in range(i + 1, len(features)):
                feat_j = features[j]
                feat_name = 'l2r_' + feat_i + '_' + feat_j
                if self.log2_values:
                    feat_value = X[feat_i] - X[feat_j]
                else:
                    feat_value = np.log2(X[feat_i]) - np.log2(X[feat_j])
                X[feat_name] = feat_value

        keep = [feat for feat in X.columns if feat.startswith("l2r_")]
        X = X[keep]

        if self.top is not None:
            X = FeatureConstruction._top(
              X, self.top, y=self.y, order_by=self.order_by)

        return X

    def build(self, methods=[], **kwargs):

        for m in methods:
            if m == "ratio":
                self._ratio(**kwargs)

        return self.X
