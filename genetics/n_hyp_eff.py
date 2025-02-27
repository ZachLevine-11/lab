import numpy as np
import pandas as pd


##we need all the data to be pre-corrected and saved for this to work
def get_n_hyp_eff(loader_dict):
    fundict = {}
    for baseSaveName in loader_dict.keys():
        fundict[baseSaveName] = pd.read_csv("/net/mraid20/export/jafar/Zach/" + baseSaveName)
        print("combined with " + str(baseSaveName))
    all_loaders = pd.concat(fundict.values(), axis=1,
                            join="inner")  ##all loaders joined by index, for each person this has all their loader data
    all_loaders_corr = all_loaders.corr()
    all_loaders_corr_square = np.square(all_loaders_corr).to_numpy()
    ##isnan indexing will only work after we convert to a single dimensional array
    all_loaders_corr_square = all_loaders_corr_square[~np.isnan(all_loaders_corr_square)]
    sst = all_loaders_corr_square.sum()
    n = len(all_loaders.columns)
    n_eff_hyp = (n ** 2) / sst
    return n_eff_hyp
