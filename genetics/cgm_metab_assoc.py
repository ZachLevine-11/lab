import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from run_gwas import read_loader_in
from LabData.DataLoaders.CGMLoader import CGMLoader
from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
from GeneticsPipeline.helpers_genetic import read_status_table
from statsmodels.stats.multitest import multipletests
from scores_work import stack_matrices_and_bonferonni_correct
from LabData.DataLoaders.PRSLoader import PRSLoader
from loop_generate_prs_matrix import loop_generate_prs_matrix

def do_assoc(status_table, merged, direction = False):
    covars = pd.read_csv("/net/mraid20/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt",sep="\t",engine="python")
    covars["RegistrationCode"] = covars["IID"].apply(
        status_table.set_index("gencove_id").RegistrationCode.to_dict().get)
    covars = covars.drop("IID", axis=1).set_index("RegistrationCode", drop=True)
    operative_cols = [x for x in merged.columns if x not in ["MODD", "SdHHMM"]]
    pvals = {"MODD" : {}, "SdHHMM" :{}}
    i = 0
    for var in ["MODD", "SdHHMM"]:
        for metab in operative_cols:
            merged_thisvar_this_metab = merged.dropna(axis=0, subset=[var, metab]).merge(covars,
                                                               left_index=True,
                                                               right_index=True,
                                                               how="inner")
            merged_thisvar_this_metab["y"] = merged_thisvar_this_metab[var]
            merged_thisvar_this_metab["metab"] = merged_thisvar_this_metab[metab] ##don't pass the actual metabolite name to the formula
            formula = "y ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + gender + age + metab"
            try:
                model = ols(formula, merged_thisvar_this_metab).fit()
                if direction:
                    pvals[var][metab] = model.params["metab"]
                else:
                    hypotheses = "(metab = 0)"
                    test_ = model.f_test(hypotheses)
                    pval = float(test_.pvalue)
                    pvals[var][metab] = pval
            except Exception as e:
                pvals[var][metab] = None
            if i/(2*len(operative_cols)) %2 == 0:
                print(i/(2*len(operative_cols)), "% done.")
            i += 1
    pvals_df = pd.DataFrame(pvals)
    if direction:
        return pvals_df
    else:
        pvals_df = pvals_df.fillna(1)
        pvals_df_corrected = pd.DataFrame(
            multipletests(pvals=pvals_df.to_numpy().flatten(), method="bonferroni")[1].reshape(pvals_df.shape))
        pvals_df_corrected["Metabolite"] = pvals_df.index.values
        pvals_df_corrected = pvals_df_corrected.set_index("Metabolite")
        pvals_df_corrected.columns = pvals_df.columns
        pvals_df_corrected.to_csv("/net/mraid20/export/jasmine/zach/height_gwas/CGM_Metab_Corrected.csv")
        return pvals_df_corrected

def do_merge():
    cgm_mat = pd.read_csv("/net/mraid20/export/jasmine/zach/height_gwas/CGM_Metab_Corrected.csv").set_index("Metabolite")
    cgm_mat_sig = cgm_mat.loc[(cgm_mat < 0.05).any(1), :]
    metab_stackmat = stack_matrices_and_bonferonni_correct(only_metab = True)
    thedict = PRSLoader().get_data().df_columns_metadata
    thedict.index = list(map(lambda thestr: "pvalue_" + thestr, thedict.index))
    thedict = thedict.h2_description.to_dict()
    metab_stackmat = metab_stackmat.rename(dict(zip(list(metab_stackmat.columns),
                                        [thedict.get(col) for col in metab_stackmat.columns])), axis=1)
    mapper = list(map(lambda potential: type(potential) == str, metab_stackmat.columns))
    metab_stackmat = metab_stackmat.loc[:, mapper]
    metab_stackmat = metab_stackmat.reset_index().set_index("Phenotype")
    metab_stackmat.index.name = "Metabolite"
    merged = pd.merge(metab_stackmat, cgm_mat_sig, left_index = True, right_index = True, how = "inner").drop("Loader", axis = 1)
    sdhmm_sig = merged.loc[merged.SdHHMM < 0.05, :].drop(["SdHHMM", "MODD"], axis = 1)
    sdhmm_sig = sdhmm_sig.loc[(sdhmm_sig < 0.05).any(1), (sdhmm_sig < 0.05).any(0)]
    modd_sig = merged.loc[merged.MODD < 0.05, ].drop(["SdHHMM", "MODD"], axis = 1)
    modd_sig = modd_sig.loc[(modd_sig < 0.05).any(1), (modd_sig < 0.05).any(0)]
    return sdhmm_sig, modd_sig

def check_direction(sdhmm, modd, status_table):
    just_sd_prs = set(sdhmm.columns) - set(modd.columns)
    just_sd_metab = set(sdhmm.index) - set(modd.index)
    just_md_prs = set(modd.columns) - set(sdhmm.columns)
    just_md_metab = set(modd.index) - set(sdhmm.index)
    cgm = read_loader_in(CGMLoader)[["MODD", "SdHHMM"]]
    metab = read_loader_in(SerumMetabolomicsLoader)[list(just_sd_metab) + list(just_md_metab)]
    metab.to_csv("/net/mraid20/export/jasmine/zach/prs_associations/corrected_loaders/" + "CGM_Metab.csv")
    merged = pd.merge(cgm, metab, left_index=True, right_index=True, how="inner")
    res_metab_direction = do_assoc(status_table, merged, direction = True)
    res_prs_direction = loop_generate_prs_matrix(test = "corrected_regression", duplicate_rows="last", saveName = "CGM_Metab.csv", tailsTest="", random_shuffle_prsLoader=False, use_prsLoader=True, direction = True, onlyThesePrses=list(just_sd_prs) + list(just_md_prs))
    return res_prs_direction, res_metab_direction

##For both MODD and sdHHMM, regress on all metabolites
##Take the significant metabolites by P value after correction
##Take the significant PRSes to those metabolites
##Grab the directions of the significant associations
if __name__ == "__main__":
    redodo_assoc = True
    do_check_direction = False
    if redodo_assoc or do_check_direction:
        status_table = read_status_table()
        try:
            status_table = status_table[status_table.passed_qc].copy()
        except ValueError:  ##In case the genetics pipeline is running
            status_table = status_table.dropna()
            status_table = status_table[status_table.passed_qc].copy()
    if redodo_assoc:
        cgm = read_loader_in(CGMLoader)[["MODD", "SdHHMM"]]
        metab = read_loader_in(SerumMetabolomicsLoader)
        merged = pd.merge(cgm, metab, left_index = True, right_index = True, how = "inner")
        do_assoc(status_table, merged, direction = False)
    if do_check_direction:
        sdhmm, modd = do_merge()
        prs_direction, res_metab_direction = check_direction(sdhmm, modd, status_table)

