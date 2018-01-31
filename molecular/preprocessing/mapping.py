# -*- coding: utf-8 -*-

from bioservices import biomart
from io import StringIO
import pandas as pd


class Mapping:

  def __init__(self, host):
    self.s = biomart.BioMart(host=host)

  def ensembl_to_hgnc(self, ensembl_ids):
    # building query
    self.s.new_query()
    self.s.add_dataset_to_xml('hsapiens_gene_ensembl')
    self.s.add_attribute_to_xml('ensembl_gene_id')
    self.s.add_attribute_to_xml('hgnc_symbol')
    xml = self.s.get_xml()

    # reading mapping
    res = pd.read_csv(StringIO(self.s.query(xml)), sep='\t', header=None)
    res.columns = ['ensembl_gene_id', 'hgnc_symbol']
    res = res.dropna()
    res = res.set_index('ensembl_gene_id')
    res = res[~res.index.duplicated(keep='first')]

    return res

  def get_mito(self, org):
    ''' Get mitochondrial gene symbols for specific organism
        Parameters:
        org : organism, supports ['hsapiens', 'mmusculus']
    '''
    # building query
    self.s.new_query()
    if org == 'hsapiens':
      self.s.add_dataset_to_xml('hsapiens_gene_ensembl')
      self.s.add_attribute_to_xml('hgnc_symbol')
    elif org == 'mmusculus':
      self.s.add_dataset_to_xml('mmusculus_gene_ensembl')
      self.s.add_attribute_to_xml('mgi_symbol')
    else:
      log.write('Org not recognized')
      return
    self.s.add_attribute_to_xml('chromosome_name')
    xml = self.s.get_xml()

    # parsing mitochondrial gene symbols
    res = pd.read_csv(StringIO(self.s.query(xml)), sep='\t', header=None)
    res.columns = ['symbol', 'chromosome_name']
    res = res.dropna()
    res = res[res['chromosome_name'] == 'MT']
    res = res.set_index('symbol')
    res = res[~res.index.duplicated(keep='first')]

    return set(res.index)

