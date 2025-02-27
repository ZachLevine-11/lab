import numpy as np
import pandas as pd
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
from run_gwas import read_loader_in
from statsmodels.stats.multitest import multipletests
import sklearn
import shap

def shap_single_metab(merged, metaboliteName, mbcols):
    gutmb_df = merged.loc[:, mbcols].dropna(axis = 1) ##drop cols with missing values, outliers
    metab_series = merged[metaboliteName]
    model = sklearn.linear_model.LinearRegression().fit(gutmb_df, metab_series)
    explainer = shap.explainers.Linear(model, gutmb_df)
    shap_values = explainer(gutmb_df)
    return shap_values

if __name__ == "__main__":
    gutmb = read_loader_in(GutMBLoader)
    metab = read_loader_in(SerumMetabolomicsLoader)
    res = {}
    for metabolite in metab.columns:
        merged = pd.merge(gutmb, metab[metabolite].dropna(axis = 0), left_index = True, right_index = True)
        res[metabolite] = shap_single_metab(merged, metabolite, mbcols = gutmb.columns)

