# -*- coding: utf-8 -*-

import IPython
import json
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objs as go
import scipy.stats as ss
import seaborn as sns
import statsmodels.graphics.gofplots as smg


class Plot:

  _palette = sns.cubehelix_palette()
  _cmap = sns.cubehelix_palette(as_cmap=True)

  @classmethod
  def plotize(data, layout=None):
    ''' Plot with Plotly.js using the Plotly JSON Chart Schema
        http://help.plot.ly/json-chart-schema/ '''

    data['layout'].update(
      font=dict(family='Arial'),
      plot_bgcolor='rgb(248, 248, 252)'
    )

    if layout is None:
      layout = {}

    redata = json.loads(
      json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder))

    relayout = json.loads(
      json.dumps(layout, cls=plotly.utils.PlotlyJSONEncoder))

    bundle = {}
    bundle['application/vnd.plotly.v1+json'] = {
      'data': redata,
      'layout': relayout,
    }

    IPython.display.display(bundle, raw=True)

  @classmethod
  def linreg(cls, data, color, name):
    ''' Linear Regression plot '''

    points = go.Scatter(
      x=data.iloc[:, 0],
      y=data.iloc[:, 1],
      mode='markers',
      text=data.index.tolist(),
      marker=dict(
        color=color
      ),
      name=name
    )

    slp, intr, r, p, serr = ss.linregress(data.iloc[:, :2])
    line = slp * data.iloc[:, 0] + intr

    fit_text = f'(R2 = {r**2:0.2f}, p = {p:0.2e})'

    fit = go.Scatter(
      x=data.iloc[:, 0],
      y=line,
      mode='lines',
      marker=go.Marker(color=color),
      hoverinfo='none',
      showlegend=False
    )

    return points, fit, fit_text

  # TODO
  @classmethod
  def hmap(cls, data, title=''):
    ''' Heatmap (Plotly) '''

    hmap = go.Heatmap(
      x=data.columns,
      y=data.index,
      z=data.values,
      colorscale='Viridis'
    )

    fig = go.Figure(data=[hmap])

    fig.layout.update(
      title=title,
      # annotations=annotations,
      xaxis=dict(ticks='', side='top'),
      # ticksuffix is a workaround to add a bit of padding
      yaxis=dict(ticks='', ticksuffix='  '),
      width=700,
      height=700,
      autosize=False
    )

    cls.plotize(fig)

    return fig

  @classmethod
  def box(cls, data, x, y, ylim=None, title=None, hue=None, rotation=None):
    ''' Boxplot '''

    g = sns.factorplot(
      x=x, y=y, data=data, kind='box', aspect=1, size=4, legend_out=True,
      linewidth=0.85, hue=hue, palette=cls._palette)

    if rotation is not None:
      g.set_xticklabels(rotation=rotation)

    if ylim is not None:
      g.set(ylim=ylim)

    if title is not None:
      g.fig.suptitle(title, size='large')
      g.fig.subplots_adjust(top=0.85)

    return g.fig

  @classmethod
  def bar(cls, data, title):
    ''' Barplot '''

    x = range(data.shape[0])
    y = data.ix[:, 0]
    # yerr = data.ix[:, 1]
    ylabels = data.index

    fig = plt.figure()
    # plt.bar(x, y, yerr=yerr, align="center")
    plt.bar(x, y, align='center')
    plt.title(title)
    plt.xticks(x, ylabels, rotation=90)
    plt.xlim([-1, len(x)])

    return fig

  @classmethod
  def heatmap(cls, data, title, xlabel, ylabel):
    ''' Annotated Heatmap '''

    fig = plt.figure()
    sns.heatmap(data, annot=True, fmt='.3f', cmap=cls._cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return fig

  @classmethod
  def roc(cls, y, proba):
    ''' ROC curve '''

    fig = plt.figure(figsize=(5, 5))

    for label in np.unique(y):
      fpr, tpr, thres = roc_curve(y, proba.loc[:, label], pos_label=label)
      roc_auc = auc(fpr, tpr)
      plt.plot(
        fpr, tpr, lw=1, label=f'ROC class {label} (area = {roc_auc:.2f})')

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    return fig

  @classmethod
  def qq(cls, data, title='', datatype=''):
    ''' Quantile-Quantile plot '''

    def quantile(x, **kwargs):
      ax = plt.gca()
      smg.qqplot(x, line='s', ax=ax)

    n_cols = int(np.sqrt(data.shape[1]))
    if np.sqrt(data.shape[1]) - n_cols > 0:
      n_cols += 1

    data = pd.melt(data)
    data.columns = [data.columns[0], datatype]
    g = sns.FacetGrid(data, col=data.columns[1], col_wrap=n_cols)
    g.map(quantile, data.columns[1])

    plt.subplots_adjust(top=0.85)
    g.fig.suptitle(title)

    for ax in g.axes:
      ax.get_lines()[0].set_markerfacecolor(sns.color_palette()[0])
      ax.get_lines()[0].set_marker('.')
      ax.get_lines()[1].set_color(sns.color_palette()[0])

    return g.fig

  @classmethod
  def histplot(cls, data, x, y, hue):
    ''' Histogram
        :param data: Pandas DataFrame (observations x features) '''

    g = sns.factorplot(
      x=x, y=y, hue=hue, data=data, kind='bar', aspect=2, size=4, legend=False,
      linewidth=0.85)
    g.set_axis_labels(x, y)
    plt.legend(loc='best', title=hue)

    return g.fig
