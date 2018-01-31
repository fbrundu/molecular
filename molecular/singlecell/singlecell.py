# -*- coding: utf-8 -*-

import logging as log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import spearmanr
import seaborn as sns

from molecular.plotting import Plot
from molecular.preprocessing import Mapping

class SingleCell:

  #FIXME use cellranger library API?
  @classmethod
  def load_cellranger(cls, mtx, genes, barcodes):
    ''' Load CellRanger output
        Parameters:
        mtx : filepath for mtx matrix with sparse data
        genes : filepath for genes list corresponding to the first
          dimension of the matrix
        barcodes : filepath for barcodes list corresponding to the
          second dimension of the matrix
    '''
    # Read the matrix
    mat = mmread(mtx)
    df = pd.SparseDataFrame(mat)
    genes = pd.read_table(genes, sep='\t', header=None)
    barcodes = pd.read_table(barcodes, sep='\t', header=None)[0]
    df.index = genes[[0, 1]]
    df.index = pd.MultiIndex.from_tuples(df.index)
    df.columns = list(barcodes)

    # For the same Gene Symbol there may be duplicates. Select the one
    # with the highest median signal
    df['Median'] = df.median(axis=1)
    df = df.sort_values(by=['Median'], ascending=False, na_position='last')
    df = df.drop(columns=['Median'])
    df = df.fillna(0).to_dense()
    df = df.groupby(level=1).first()

    # Transpose in the form cells x genes
    df = df.T

    return df

  @classmethod
  def plot_exp(cls, df, top, fun='sum'):
    ''' Plot the distribution of the genes which are highly
        expressed across cells
        Parameters:
        df : pandas DataFrame cells x genes
        top : number of genes to plot
        fun : function to compute on each gene ['sum', 'median', 'mean']
    '''
    figsize = (5, (top // 20) * 6)
    if fun == 'sum':
      top = df.sum().sort_values(ascending=False)[:top].index
    elif fun == 'median':
      top = df.median().sort_values(ascending=False)[:top].index
    elif fun == 'mean':
      top = df.mean().sort_values(ascending=False)[:top].index
    else:
      log.write(f'{fun} not implemented')
      return
    fig = plt.figure(dpi=100, figsize=figsize)
    sns.boxplot(data=df[top], orient='h', fliersize=1)
    plt.xlabel('Reads count per cell')
    plt.ylabel('Top expressed genes')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

    return fig

  @classmethod
  def remove_ribo(cls, df, org):
    ''' Remove ribosomal genes from dataset
        Parameters:
        df : pandas DataFrame cells x genes
        org : organism, supports ['hsapiens', 'mmusculus']
    '''
    if org == 'hsapiens':
      regex = '^RP[SL].[0-9]*'
    elif org == 'mmusculus':
      regex = '^Rp[sl].[0-9]*'
    else:
      log.write(f'{org} organism not supported')
      return

    df = df.drop(df.filter(regex='^Rp[sl].[0-9]*').columns, axis=1)

    return df

  @classmethod
  def prepare_ratio_qc(cls, df, org):
    ''' Prepare matrix for quality control based on mitochondrial
        genes and ERCC spike-ins to filter out bad quality cells.
        Ref: Ilicic T, Kim JK, Kolodziejczyk AA, Bagger FO, McCarthy DJ, Marioni JC, Teichmann SA. Classification of low quality cells from single-cell RNA-seq data. Genome biology. 2016 Feb 17;17(1):29.)

        Parameters:
        df : pandas DataFrame cells x genes
        org : organism ['hsapiens', 'mmusculus']
    '''
    #Separate ERCC spike-ins from dataset
    if org == 'hsapiens':
      prefix = 'ERCC'
    elif org == 'mmusculus':
      prefix = 'Ercc'
    else:
      log.write(f'{org} organism not supported')
      return None, None
    ercc = df.T[df.columns.str.startswith(prefix)].T
    df = df.T[~df.columns.str.startswith(prefix)].T

    #Separate mitochondrial genes from dataset
    m = Mapping(host='useast.ensembl.org')
    mt_genes = m.get_mito(org) & set(df.columns)
    mt = df[list(mt_genes)]
    df = df.drop(list(mt_genes), axis=1)

    return df, ercc, mt

  @classmethod
  def plot_ratio_qc(cls, df, ercc, mt, filters={}, apply_filter=False):
    ''' Plot quality control based on mitochondrial genes and ERCC
        spike-ins to filter out bad quality cells.
        Ref: Ilicic T, Kim JK, Kolodziejczyk AA, Bagger FO, McCarthy DJ, Marioni JC, Teichmann SA. Classification of low quality cells from single-cell RNA-seq data. Genome biology. 2016 Feb 17;17(1):29.)

        Parameters:
        df : pandas DataFrame cells x genes
        ercc : pandas DataFrame cells x genes
        mt : pandas DataFrame cells x genes
        filters : dictionary to set filter thresholds (e.g. {'ercc': 2e-3, 'mt': .1} )
        apply_filter : to filter out bad quality cells based on filter thresholds
    '''
    ercc_plt, mt_plt = False, False
    if ercc is not None and ercc.shape[0] > 0 and ercc.shape[1] > 0:
      ercc_x, ercc_y = cls._plot_ratio_qc_compute(df, ercc)
      ercc_plt = True
    if mt is not None and mt.shape[0] > 0 and mt.shape[1] > 0:
      mt_x, mt_y = cls._plot_ratio_qc_compute(df, mt)
      mt_plt = True

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(10, 5))
    hq_cells = df.index

    if ercc_plt:
      ax1.scatter(ercc_x, ercc_y, marker='.', s=1, c='k', alpha=.5)
      plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
      ax1.set_xlabel('Total number of detected (nonzero) genes')
      ax1.set_ylabel('ERCC transcripts percentage')
      if 'ercc' in filters:
        ax1.axhline(y=filters['ercc'], linewidth=1, color='k')
        if apply_filter:
          hq_cells &= df[ercc_y < filters['ercc']].index

    if mt_plt:
      ax2.scatter(mt_x, mt_y, marker='.', s=1, c='k', alpha=.5)
      plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
      ax2.set_xlabel('Total number of detected (nonzero) genes')
      ax2.set_ylabel('Mitochondrial transcripts percentage')
      if 'mt' in filters:
        ax2.axhline(y=filters['mt'], linewidth=1, color='k')
        if apply_filter:
          hq_cells &= df[mt_y < filters['mt']].index

    df = df.loc[hq_cells]

    return fig, df

  @classmethod
  def _plot_ratio_qc_compute(cls, df, ctrl):
    nonzero_features = (df > 0).sum(axis=1)
    counts_endo = df.sum(axis=1)
    counts_ctrl = ctrl.sum(axis=1).loc[counts_endo.index]
    return nonzero_features, counts_ctrl / counts_endo

  @classmethod
  def plot_dist_qc(cls, df, filters={}, apply_filter=False):
    '''
    '''
    hq_cells = df.index

    x = df.astype(bool).sum(axis=1)
    y = np.log10(df.sum(axis=1) + 1)

    g = sns.jointplot(x, y, size=9, stat_func=spearmanr,
        joint_kws={'s':1, 'color': 'k', 'alpha':.3},
        marginal_kws={'bins':200, 'color': 'k'})
    g.set_axis_labels('Number of detected genes', 'Library size (log10)')
    g.ax_joint.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    g.ax_joint.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    if 'libsize' in filters:
      g.ax_joint.axhline(y=filters['libsize'], linewidth=1, color='k', alpha=.8)
      if apply_filter:
        hq_cells &= df[y > filters['libsize']].index

    if 'detected' in filters:
      g.ax_joint.axvline(x=filters['detected'], linewidth=1, color='k', alpha=.8)
      if apply_filter:
        hq_cells &= df[x > filters['detected']].index

    df = df.loc[hq_cells]

    return g.fig, df

  @classmethod
  def clean_low_exp(cls, df, thr=None, apply_filter=False):
    '''
    '''
    #Clean never expressed genes
    df = df.T[(df.sum(axis=0) > 0)].T

    hq_genes = df.columns

    fig = plt.figure(dpi=100)
    sns.distplot(np.log10(df.mean(axis=0)), kde=False, color='k', bins=200)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    plt.xlabel('Average expression (log10)')
    plt.ylabel('Number of genes')

    if thr is not None:
      plt.axvline(x=thr, linewidth=1, color='k', alpha=.8)
      if apply_filter:
        hq_genes &= df.T[np.log10(df.mean(axis=0)) > thr].T.columns

    df = df.loc[:, hq_genes]

    return fig, df

