# -*- coding: utf-8 -*-

import logging as log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread
import seaborn as sns

from molecular.plotting import Plot

class SingleCell:

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
  def plot_most_exp_genes(cls, df, top):
    ''' Plot the distribution of the genes which are highly
        expressed across cells
        Parameters:
        df : pandas DataFrame cells x genes
        top : number of genes to plot
    '''
    figsize = (5, (top // 20) * 6)
    top = df.sum().sort_values(ascending=False)[:top].index
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
      log.write(f'{org} not supported')
      return

    df = df.drop(df.filter(regex='^Rp[sl].[0-9]*').columns, axis=1)

    return df

