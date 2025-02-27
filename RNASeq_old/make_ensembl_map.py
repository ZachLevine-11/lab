##run in python3 console, not in conda kernel as we can't load this package into it.
##the biomart package is installed via pip, so the import works with the default interpreter but not the conda one
##Pycharm will say that the biomart package doesn't exist because it uses the conda interpreter
##run this file from the shell

##to run, from shell: python3 ~/PycharmProjects/RNASeq/make_ensembl_map.py
import numpy as np
import pandas as pd
import biomart

##map human ensembl gene ids to human gene names
#adapted from from https://gist.github.com/ben-heil/cffbebf8865795fe2efbbfec041da969
def make_ensembl_id_map():
    server = biomart.BiomartServer('http://asia.ensembl.org/biomart') #useast
    mart = server.datasets['hsapiens_gene_ensembl']
    attributes = ['ensembl_gene_id', 'hgnc_symbol']
    response = mart.search({'attributes': attributes})
    data = response.raw.data.decode('ascii')
    ensembl_to_genesymbol = {}
    for line in data.splitlines():
        line = line.split('\t')
        # The entries are in the same order as in `attributes`
        ensembl_to_genesymbol[line[0]] = line[1]
    return ensembl_to_genesymbol

themap = make_ensembl_id_map()
pd.Series(themap).to_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/RNASeq/ensembl_map.csv")