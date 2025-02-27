import matplotlib.pyplot as plt
from run_gwas import read_original_plink_bins, read_loader_in
from pandas_plink import read_plink1_bin
import numpy as np
import pandas as pd
from GeneticsPipeline.helpers_genetic import read_status_table
import torch
from torch.utils.data import Dataset
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.UltrasoundLoader import UltrasoundLoader
from LabData.DataLoaders.DietLoggingLoader import DietLoggingLoader
from LabData.DataLoaders.ABILoader import ABILoader
from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
from LabData.DataLoaders.ItamarSleepLoader import ItamarSleepLoader
from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.LifeStyleLoader import LifeStyleLoader
from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
from LabData.DataLoaders.QuestionnairesLoader import QuestionnairesLoader
from LabData.DataLoaders.HormonalStatusLoader import HormonalStatusLoader
from LabData.DataLoaders.IBSTenkLoader import IBSTenkLoader
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataLoaders.ChildrenLoader import ChildrenLoader
from LabData.DataLoaders.CGMLoader import CGMLoader
from statsmodels.formula.api import ols


##returns the original 10K bins with the 10K index
def read_plink_bins_10K_index():
    status_table = read_status_table()
    try:
        status_table = status_table[status_table.passed_qc].copy()
    except ValueError:  ##In case the genetics pipeline is running
        status_table = status_table.dropna()
        status_table = status_table[status_table.passed_qc].copy()
    unique_bin = read_original_plink_bins()
    unique_bin["sample"] = unique_bin.sample.to_pandas().apply(
        status_table.set_index('gencove_id').RegistrationCode.to_dict().get)
    return (unique_bin)


##repeat a plink analysis for a single snp
# Pheno is the series of the covariate we want
def manual_assoc_test(boxplot, SNPdf, pheno):
    covars = pd.read_csv("/net/mraid20/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt", sep="\t")
    covars["RegistrationCode"] = covars["IID"].apply(
        status_table.set_index("gencove_id").RegistrationCode.to_dict().get)
    covars = covars.drop("IID", axis=1).set_index("RegistrationCode", drop=True)
    fancydf = SNPdf.reset_index("variant", drop=True).merge(pheno, left_index=True, right_index=True,
                                                            how="inner").dropna(axis=0).merge(covars, left_index=True,
                                                                                              right_index=True,
                                                                                              how="inner")
    fancydf = fancydf[
        ["genotype", pheno.name, "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "age",
         "gender"]]
    fancydf["y"] = fancydf[pheno.name]  ##fit with a "y", so the pheno name doesn't break the parsing of the formula
    ##save original labels before variance standardization for plotting
    original_geno_labels = fancydf["genotype"]
    ##equilvalent to --variance standardize, zero mean and unit variance everything we model
    fancydf = (fancydf - fancydf.mean()) / fancydf.std(
        ddof=0)  ##dd0f is default 1 in pandas and 0 in numpy and in plink, allign them
    if boxplot:
        pheno_genotype_0 = fancydf.loc[original_geno_labels == 0, "y"]
        pheno_genotype_1 = fancydf.loc[original_geno_labels == 1, "y"]
        pheno_genotype_2 = fancydf.loc[original_geno_labels == 2, "y"]
        ##this dataframe is not alligned, it just holds the distribution of phenotypes for each genotype class
        plotting_df = pd.concat([pheno_genotype_0, pheno_genotype_1, pheno_genotype_2], axis=1)
        plotting_df.columns = ["minor-minor", "minor-major", "major-major"]
        plotting_df.boxplot()
    formula = " y ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + gender + age + genotype"
    try:
        model = ols(formula, fancydf).fit()
        if boxplot:
            formula_without = " y ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + gender + age"
            plt.title("r^2 with =" + str(model.rsquared)[0:5] + ", r^2 without = " + str(
                ols(formula_without, fancydf).fit().rsquared)[0:5])
            plt.show()
        hypotheses = "(genotype = 0)"
        test_ = model.wald_test(hypotheses)
        pval = float(test_.pvalue)
        return pval
    except Exception as e:
        return None

##for a given phenotype, compare the pvalues for its top hits in gwas to those from a "manual gwas"
def manual_associate_top_k_hits_compare(boxplot=True, var='pregnancy_end_reason_[JA62]', loader=ChildrenLoader,
                                        plink_data=None,
                                        gwas_results_renamed_dir="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_results_renamed/"):
    manual_res = {}
    plink_res = {}
    loader = read_loader_in(loader)
    phenotype = loader[var]
    thefilename = gwas_results_renamed_dir + var + ".glm.linear"
    read_in_gwas_res = pd.read_csv(thefilename, sep="\t")
    if len(read_in_gwas_res.P) != 0:  # ignore gwases with no significant associations and do FDR
        read_in_gwas_res.columns = ["CHROM", *read_in_gwas_res.columns[range(1, len(read_in_gwas_res.columns))]]
        read_in_gwas_res = read_in_gwas_res.sort_values(by="P", axis=0)
        ##Probably an overly conservative correction because we don't GWAS every phenotype in a loader, but for now assume that we do
        top_plink_results = read_in_gwas_res.loc[read_in_gwas_res.P < (5 * 10 ** (-8)) / len(loader.columns), :]
        if top_plink_results.shape[0] == 0:
            print("No significant hits, nothing to test")
            return None
        print("There are " + str(top_plink_results.shape[0]) + " hits to test. Starting now.")
        for rsid in top_plink_results.ID:
            snp_df = plink_data.loc[:, plink_data.snp == rsid].to_dataframe().drop("gender",
                                                                                   axis=1)  # the #gender column is meaningless here
            manual_res[rsid] = manual_assoc_test(boxplot, snp_df, phenotype)
            plink_res[rsid] = top_plink_results.loc[top_plink_results.ID == rsid, "P"].values[
                0]  ##drop the row index and the array dimension
    plink_res = pd.Series(plink_res)
    manual_res = pd.Series(manual_res)
    plink_res.name = "plink"
    manual_res.name = "manual"
    resdf = pd.DataFrame(plink_res).merge(manual_res, left_index=True, right_index=True)
    return resdf


if __name__ == "__main__":
    plink_bins_loaded = read_plink_bins_10K_index(unique=False)
    status_table = read_status_table()
    status_table = status_table[status_table.passed_qc].copy()
    # manual_associate_top_k_hits_compare(boxplot = True, var = "fBin__55|gBin__210|sBin__304", loader = GutMBLoader, plink_data = plink_bins_loaded)
