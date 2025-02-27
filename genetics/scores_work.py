import numpy as np
import pandas as pd
from os.path import isfile, join
from GeneticsPipeline.helpers_genetic import read_status_table
from run_gwas import loaders_list, read_loader_in, update_covariates, pre_filter, summarize_gwas
from manual_gwas import read_plink_bins_10K_index
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from LabUtils.addloglevels import sethandlers
# from GeneticsPipeline.config import gencove_logs_path
import os
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns
##Need these imported for the loader_assoc_plot
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.UltrasoundLoader import UltrasoundLoader
from LabData.DataLoaders.ABILoader import ABILoader
from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
from LabData.DataLoaders.ItamarSleepLoader import ItamarSleepLoader
from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
from LabData.DataLoaders.HormonalStatusLoader import HormonalStatusLoader
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from LabData.DataLoaders.CGMLoader import CGMLoader
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.PRSLoader import PRSLoader
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols, logit
from GeneticsPipeline.helpers_genetic import read_status_table
import statsmodels.api as sm

subs = SubjectLoader().get_data().df.reset_index()
gender_map = subs.set_index("RegistrationCode")
gender_map = gender_map.loc[~gender_map.index.duplicated(), "gender"].dropna().astype(int).to_dict()

covars = pd.read_csv("/net/mraid20/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt",
                     sep="\t",
                     engine="python").drop(list(map(lambda x: "PC" + str(x), list(range(1, 11)))),
                                           axis=1)  # if using plink1 covariates, need to .drop("FID", axis=1)
status_table = read_status_table()
try:
    status_table = status_table[status_table.passed_qc].copy()
except ValueError:  ##In case the genetics pipeline is running
    status_table = status_table.dropna()
    status_table = status_table[status_table.passed_qc].copy()
covars["RegistrationCode"] = covars["IID"].apply(
    status_table.set_index("gencove_id").RegistrationCode.to_dict().get)
covars = covars.drop("IID", axis=1).set_index("RegistrationCode", drop=True)

##test can be "t" for t test, or "r" for regression
##t testing expects an index of PRS_class = 0,1,2
##takes a single batch and computes a test for people split into high and lowprs for each column
##regression expects the index to be the prs itself
##the swap argument swaps y and x after data selection, which is useful for inverting the model, and is only supported for test = "r"
def manyTestsbatched(batch, test, tailsTest, direction = False, gender = 2):
    pd.options.mode.use_inf_as_na = True  ##treat inf as na values to save number of checks
    batchtypes = dict(batch.dtypes)

    pvals = {}
    ##fix numberofwakes type errors with duplicate columns
    ##the value at index var corresponds to whether var is a repeated column or not.
    duplicated_cols = pd.DataFrame(batch.columns.duplicated(), index=batch.columns)
    # exclude missing data by default
    operative_cols = [x for x in batch.columns if x not in ["prs"]]
    for var in operative_cols:  ##for corrected_regression the index is 10K so prs is a column, ignore it and use it separately
        ##if the column is not repeated. We have to do this before indexing the columns to avoid indexing errors.
        if sum(duplicated_cols[duplicated_cols.index == var][0]) == 0:
            ## if the data is non numeric or entirely missing, skip
            if test == "t" or test == "m":
                should_skip = batchtypes[var] != "float64" \
                              or batch[batch.index == 1][var].isna().sum() == len(batch[batch.index == 1]) \
                              or batch[batch.index == 2][var].isna().sum() == len(batch[batch.index == 2])
            else:
                ##the condition is different because for regression, we don't have the prs classes
                should_skip = (batchtypes[var] != "float64" and batchtypes[var] != "int64") or batch[var].isna().sum() == len(batch)
            if should_skip:
                pvals[var] = None
            else:
                ##we have a numeric variable, so run tests
                ##w're indexing by columns, we always want the same rows
                if test == "t":
                    if tailsTest == "rightLeft":
                        test_res = ttest_ind(batch[batch.index == 1][var].dropna(),
                                             batch[batch.index == 2][var].dropna())
                    else:
                        test_res = ttest_ind(batch[batch.index == 2][var].dropna(),
                                             batch[batch.index != 2][var].dropna())
                    pvals[var] = test_res[1]
                elif test == "m":
                    if tailsTest == "rightLeft":
                        test_res = mannwhitneyu(batch[batch.index == 1][var].dropna(),
                                                batch[batch.index == 2][var].dropna())
                    else:
                        test_res = mannwhitneyu(batch[batch.index == 2][var].dropna(),
                                                batch[batch.index != 2][var].dropna())
                    pvals[var] = float(test_res.pvalue)
                elif test == "corrected_regression":
                    batch_droppedna_this_var = batch.dropna(axis=0, subset=[var]).merge(covars,
                                                                                        left_index=True,
                                                                                        right_index=True,
                                                                                        how="inner")
                    ##special characters in any variable names breaks hypothesis testing, so do the testing with a different name and then save under the original one
                    batch_droppedna_this_var["y"] = batch_droppedna_this_var[var]
                    batch_droppedna_this_var = batch_droppedna_this_var.loc[~batch_droppedna_this_var.index.duplicated(),:]
                    ##Use a formula so we can access these names in hypothesis testing later
                    ##test whether the PRS coefficient is different from zero
                    if gender != 2:
                        batch_droppedna_this_var = batch_droppedna_this_var.loc[list(map(lambda x: gender_map.get(x) == gender, batch_droppedna_this_var.index)),:]
                    try:
                        if len(batch_droppedna_this_var["y"].unique()) > 2:
                            formula = "y ~ age + prs + const"
                            model = ols(formula, sm.add_constant(batch_droppedna_this_var)).fit()
                        else:
                            formula = "y ~ age + prs"
                            model = logit(formula, batch_droppedna_this_var).fit()
                        if direction:
                            pvals[var] = model.params["prs"]
                        else:
                            ##remember that we need the formula-based version of this function
                            hypotheses = "(prs = 0)"
                            test_ = model.f_test(hypotheses)
                            pval = float(test_.pvalue)
                            pvals[var] = pval
                    except Exception as e:  ##catch hypothesis testing errors
                        pvals[var] = None
        else:  ##add None the same time as the number of repeats to allign the indexes with column names of the batch
            NumberofNones = sum(duplicated_cols[duplicated_cols.index == var][0]) + 1
            for NoneNumber in range(NumberofNones):
                p_label = var + "_" + str(NoneNumber)  ##avoid duplicate column names in the output
                pvals[p_label] = None
    return pd.Series(pvals)


def get_all_result_files(SOMAscan=True,
                         cooldir="/net/mraid20/export/jasmine/zach/scores/score_results/"):
    if SOMAscan: allDir = cooldir + "SOMAscan/"
    onlyfiles = [f for f in os.listdir(allDir) if isfile(join(allDir, f)) and f.endswith(".sscore")]
    return onlyfiles


##combine all the raw score files and read them, saving the result
def combine_scores(SOMAscan=True,
                   cooldir="/net/mraid20/export/jasmine/zach/scores/score_results/"):
    i = 0
    all_score_files = get_all_result_files(SOMAscan=SOMAscan, cooldir=cooldir)
    numFiles = len(all_score_files)
    ##merge on Gencove ID (IID), populating the id list from a random results file first
    all_df = pd.read_csv(cooldir + "SOMAscan/" + "GSTP1.4911.49.2_model.txt.sscore", sep="\t").iloc[:, 0]
    all_df.name = "IID"
    for fileName in all_score_files:
        print("Now reading in: " + str(fileName) + ", which is: " + str(i) + "/" + str(numFiles))
        newdf = pd.read_csv(cooldir + "SOMAscan/" + fileName, sep="\t")
        newdf.columns = ["IID", fileName.split('_')[0]]
        all_df = pd.merge(all_df, newdf, left_on="IID", right_on="IID", how="inner")
        i += 1
    all_df['RegistrationCode'] = all_df['IID'].apply(
        status_table.set_index('gencove_id').RegistrationCode.to_dict().get)
    all_df = all_df.set_index("RegistrationCode").drop("IID", axis=1)
    all_df.to_csv(raw_qtl_fname)
    return all_df


###after calling combined_scores once, you can use this function to get at the saved result without having to wait for combine_scores to run again
def read_saved_combined_scores(fname):
    return pd.read_csv(fname).set_index("RegistrationCode")


def correct_all_loaders(loaders=None, correct_beforehand=False, plink_data=None, most_frequent_val_max_freq=0.95,
                        min_subject_threshold=3000):
    for loader in loaders:
        operative_loader = read_loader_in(loader)
        operative_loader = pre_filter(operative_loader, plink_data=plink_data,
                                      most_frequent_val_max_freq=most_frequent_val_max_freq,
                                      min_subject_threshold=min_subject_threshold)
        justname = str(loader).split(".")[2] + ".csv"
        saveName = corrected_loader_save_path + justname
        if correct_beforehand:
            pass
            print("Wrote corrected: " + saveName)
        else:
            operative_loader.to_csv(saveName)
            print("Wrote uncorrected: " + saveName)

def q_generate_prs_matrix(test="m", duplicate_rows="mean", saveName=None, tailsTest="rightLeft",
                          random_shuffle_prsLoader=False, which="RNA", direction = False, onlyThesePrses = None, gender = 2, Y_linked_genes = []):
    os.chdir("/net/mraid20/export/mb/logs/")
    # sethandlers()
    ## create the qp before doing anything with big variables, and delete everything that isn't required before calling qp
    if which == "RNA":
        y_df = pd.read_csv("/net/mraid20/export/jasmine/zach/prs_associations/corrected_loaders/" + saveName.split("_")[0] + ".csv").rename({"Unnamed: 0": "RegistrationCode"}, axis=1).set_index("RegistrationCode")
        rna_df = pd.read_pickle("/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/rna_options_final/" + "after_batch_correction_filtered_1000_no_regress_5mln_sample.df")
        gene_names = list(rna_df.columns)
        rna_df = np.log1p(np.clip(rna_df, -0.999, None, ))
        if gender == 0:
            rna_df = rna_df.drop(Y_linked_genes, axis = 1)
        fundict = {}
        ###We also care about the column names
        whole_df = pd.merge(rna_df, y_df, left_index=True, right_index = True, how="inner")
        for id in range(len(rna_df.columns)): #
            print("now onto gene: ", id)
            gene = rna_df.columns[id]
            try:
                fundict[gene] = manyTestsbatched(whole_df.rename({gene: "prs"}, axis = 1)[["prs"] + list(y_df.columns)],
                                 test,
                                 tailsTest,
                                 direction, gender)
            except Exception:
                fundict[gene] = None
        for k, v in fundict.copy().items():  ##catch broken PRSes, don't iterate over original dictionary
            if v is None:
                del fundict[k]
        try:
            return pd.DataFrame(fundict)
        except ValueError:
            return None

def make_test_all_loaders(loaders=None, loop=False, which="PQTLS", test="corrected_regression", direction = False, genders = [0, 1, 2], Y_linked_genes = []):
    for loader in loaders:
        for gender in genders:
            justname = str(loader).split(".")[2] + "_" + str(gender) + "_" + ".csv"
            res_m_loader = q_generate_prs_matrix(test=test, duplicate_rows="last", saveName=justname, tailsTest=None,
                                         random_shuffle_prsLoader=False, which=which, direction = direction,
                                                 gender = gender, Y_linked_genes = Y_linked_genes)
            if which == "RNA":
                res_m_loader.to_csv(raw_matrices_save_path_rna + justname)
                print("Wrote: " + raw_matrices_save_path_rna + justname)
            elif which == "PQTLS":
                res_m_loader.to_csv(raw_matrices_save_path_pqtl + justname)
                print("Wrote: " + raw_matrices_save_path_pqtl + justname)
            else:
                res_m_loader.to_csv(raw_matrices_save_path_prs + justname)
                print("Wrote: " + raw_matrices_save_path_prs + justname)


def stack_matrices_and_bonferonni_correct(results_dir, fillwithNA=True, orderbySig=False,
                                          include_mb=False,
                                          include_metab = False,
                                          only_metab = False):
    if include_mb and include_metab:
        all_results_files = [f for f in os.listdir(results_dir) if isfile(join(results_dir, f))]
    elif not include_mb and not include_metab:
        all_results_files = [f for f in os.listdir(results_dir) if isfile(
            join(results_dir, f)) and "GutMBLoader" not in f and "SerumMetabolomicsLoader" not in f]
    elif include_mb and not include_metab:
        all_results_files = [f for f in os.listdir(results_dir) if isfile(
            join(results_dir, f)) and "SerumMetabolomicsLoader" not in f]
    else:
        all_results_files = [f for f in os.listdir(results_dir) if isfile(
            join(results_dir, f)) and "GutMBLoader" not in f]
    if only_metab:
        all_results_files = ["SerumMetabolomicsLoader.csv"]
    dfs = []
    loader_col = []
    for res_file in all_results_files:
        tempdf = pd.read_csv(results_dir + res_file).set_index("Unnamed: 0")
        loader_col += list(np.repeat(res_file.split(".csv")[0], len(tempdf)))  ##we started with one element
        dfs.append(tempdf)
    res = pd.concat(dfs).fillna(1)
    ##we want to correct for all tests, so unwrap dataframe to be 1d then rewrap after
    ##corrected p values stay in, so this works in original order
    res_corrected = pd.DataFrame(
        multipletests(pvals=res.to_numpy().flatten(), method="fdr_bh")[1].reshape(res.shape))
    res_corrected["Phenotype"] = res.index.values
    res_corrected["Loader"] = loader_col
    res_corrected = res_corrected.set_index(["Phenotype", "Loader"])
    res_corrected.columns = res.columns
    if fillwithNA:
        res_corrected = res_corrected.mask(res_corrected > 0.05, np.nan)  ##only store sig assocs
        if orderbySig:  ##very computationally costly
            sigMap = {}
            for pheno, loader in res_corrected.index:
                sigMap[(pheno, loader)] = res_corrected.loc[res_corrected.index.get_level_values(0) == pheno,
                                          :].isna().sum().sum()
            return res_corrected.loc[pd.Series(sigMap).sort_values().to_dict().keys(),
                   :]  ##return rows (phenotypes) sorted in order of most to least significant associations
        else:
            return res_corrected
    else:
        return res_corrected

def loader_assoc_plot(stacked_mat):
    loader_assoc_map = {}
    potential_loaders = list(set(stacked_mat.index.get_level_values(1)))
    for loader in potential_loaders:
        justloader = stacked_mat.loc[stacked_mat.index.get_level_values(1) == loader, :]
        justloader_flat = justloader.to_numpy().flatten()
        loader_assoc_map[loader] = len(justloader_flat[justloader_flat < 0.05])
    res = pd.Series(loader_assoc_map)
    backupindex = res.index
    res.index = list(map(lambda name: name.split("Loader")[0], res.index.values))  ##shorten the names so they fit
    ##map loader names to loader instances to read in and normalize by number of phenotypes
    numEachLoader = list(map(lambda loaderName: len(read_loader_in(eval(loaderName))), backupindex))
    plt.bar(res.index, res.values)
    plt.yscale("log")
    plt.xticks(rotation=45, fontsize=6, ha="right")
    plt.title("Significant Associations by Loader, Raw")
    plt.show()
    plt.bar(res.index, np.divide(res.values, numEachLoader))
    plt.yscale("log")
    plt.xticks(rotation=45, fontsize=6, ha="right")
    plt.title("Significant Associations by Loader, Normalized")
    plt.show()

def make_top_sig_assocs_table(stackmat, n):
    pvals = []
    traits = []
    prses = []
    thedict = PRSLoader().get_data().df_columns_metadata.h2_description.to_dict()
    stackmat = stackmat.loc[(stackmat < 0.05).any(1), (stackmat < 0.05).any(0)]
    for i in range(len(stackmat)):
        for j in range(len(stackmat.columns)):
            if stackmat.iloc[i, j] < 0.05:
                traits.append(stackmat.index[i][0])
                prses.append(thedict[stackmat.columns[j].split("pvalue_")[-1]])
                pvals.append(stackmat.iloc[i, j])
    ans = pd.concat([pd.Series(traits),pd.Series(prses), pd.Series(pvals)], axis = 1)
    ans.columns = ["10K Trait", "UKBB PRS", "P"]
    ans = ans.dropna(subset = ["UKBB PRS"]) ##Drop associations if the PRS has no interpretable meaning
    ans = ans.sort_values("P").iloc[0:n, :]
    ans.to_csv("~/Desktop/PRS_assocs_sig.csv")
    return ans

##Stackmat should have fillNa = False
def make_clustermaps(stackmat):
    thedict = PRSLoader().get_data().df_columns_metadata
    thedict.index = list(map(lambda thestr: "pvalue_" + thestr, thedict.index))
    thedict = thedict.h2_description.to_dict()
    stackmat = stackmat.rename(dict(zip(list(stackmat.columns),
                                        [thedict.get(col) for col in stackmat.columns])),axis=1)
    mapper = list(map(lambda potential: type(potential) == str, stackmat.columns))
    stackmat = stackmat.loc[:, mapper]
    for theval in stackmat.index.get_level_values(1).unique():
        if theval == "DEXALoader":
            sns.set(font_scale=1.3)
        else:
            sns.set(font_scale=1.15)
        print("Now generating ", theval)
        s_sig = stackmat.loc[stackmat.index.get_level_values(1) == theval, :]
        s_sig.index = s_sig.index.get_level_values(0)
        if theval == "DEXALoader":
            s_sig.columns = list(map(lambda thestr: thestr[0:30], s_sig.columns))
        elif theval == "CGMLoader":
            variability_appendage_dict = dict(zip(s_sig.index, s_sig.index))
            variability_appendage_dict["mage"] = "mage" + " (WD)"
            variability_appendage_dict["SdW"] = "SdW"  + " (WD)"
            variability_appendage_dict["SdHHMM"] = "SdHHMM" + " (WD)"
            variability_appendage_dict["MODD"] = "MODD" + " (BD)"
            variability_appendage_dict["SdDM"] = "SdDM" + " (BD)"
            variability_appendage_dict["Conga"] = "Conga" + " (BD)"
            s_sig.index = list(map(lambda thestr: variability_appendage_dict[thestr], s_sig.index))
        s_sig = s_sig.loc[(s_sig < 0.05).any(1), (s_sig < 0.05).any(0)]
        if theval != "DEXALoader": #for DEXA don't label all the phentyoes
            sns.clustermap(-np.log10(s_sig),
                           cmap="Blues",
                           yticklabels=True,
                           vmax = 10)
        else:
            sns.clustermap(-np.log10(s_sig),
                           cmap="Blues",
                           vmax = 100)


##print PRSES clusters from all clusters
def prstoGwas(stackmat, threshold=0.7, do_corr=False, onlyMultiloaders=False):
    df = stackmat.loc[:, stackmat.ne(1).any()]
    df = df[df.ne(1).any(axis=1)]
    df_log = np.log(df).fillna(1)
    if do_corr:
        corr = df.corr('pearson')
        ##handle tiny numbers with integer overflow
        link = linkage(squareform(np.clip(1 - corr, a_min=0, a_max=None)), method='average')
    else:
        link = linkage(df, method="average")
    dn = dendrogram(link, no_plot=True)
    clst = fcluster(link, criterion='distance', t=threshold)
    if do_corr:
        clust_col_identity = pd.Series(index=corr.columns, data=clst).iloc[
            dn['leaves']]  ##for each PRS or phenotype, gives which cluster it's in
    else:
        clust_col_identity = pd.Series(index=df.index, data=clst).iloc[dn['leaves']]
    for identity in range(clust_col_identity.max()):
        if len(clust_col_identity[clust_col_identity.eq(identity)]) > 1:  ##only print clusters with size   > 1
            phenos, loaders = clust_col_identity[clust_col_identity.eq(identity)].index.get_level_values(0), \
                              clust_col_identity[clust_col_identity.eq(identity)].index.get_level_values(1)
            if len(set(loaders)) > 1 or not onlyMultiloaders:
                print("Found multi-loader cluster of, ", phenos, " from the loaders :", set(loaders))
                try:
                    print("Trying GWAS hit intersection")
                    summarize_gwas(phenos, use_clumped=True)
                except TypeError:
                    print("No sig hits for any of these phenotypes")
            else:
                print("Skipping single-loader cluster of, ", phenos)


def report_pheno(pheno, descriptionmap, stacked):
    print("Phenotype: ", pheno, ", Loader: ",
          stacked.index.get_level_values(1).values[stacked.index.get_level_values(0) == pheno][0])
    for prsentry in stacked.loc[pheno, :].T.dropna().index:
        print(descriptionmap[prsentry.split("pvalue_")[1]], ": ", stacked.loc[pheno, :].T.dropna().loc[prsentry].values)
    print("---------------------------------------------------------------")


if __name__ == "__main__":
    Y_linked_genes =["NLGN4Y", "UTY", "USP9Y", "KDM5D", "DDX3Y", "ZFY", "RPS4Y1", "EIF1AY", "TMSB4Y", "PRKY", "RPS4Y2"]
    ##to remove circular dependencies, these are hardcoded everywhere
    # if you want to change these, you're going to need to change these everywhere in the code that they are used, i.e preprocess_data_loader
    ##only load the status table once and pass it around to save on memory
    status_table = read_status_table()
    try:
        status_table = status_table[status_table.passed_qc].copy()
    except ValueError:  ##In case the genetics pipeline is running
        status_table = status_table.dropna()
        status_table = status_table[status_table.passed_qc].copy()
    raw_qtl_fname = "/net/mraid20/export/jasmine/zach/scores/score_results/SOMAscan/scores_all_raw.csv"
    corrected_qtl_fname_base = "/net/mraid20/export/jasmine/zach/prs_associations/corrected_loaders/"
    corrected_qtl_savename = "scores_all_corrected.csv"
    corrected_qtl_fname = corrected_qtl_fname_base + corrected_qtl_savename
    corrected_loader_save_path = "/net/mraid20/export/jasmine/zach/prs_associations/corrected_loaders/"
    raw_matrices_save_path_prs = "/net/mraid20/export/jasmine/zach/prs_associations/uncorrected_matrices_prses/"
    raw_matrices_save_path_pqtl = "/net/mraid20/export/jasmine/zach/prs_associations/uncorrected_matrices_pqtls/"
    raw_matrices_save_path_pqtl = "/net/mraid20/export/jasmine/zach/prs_associations/uncorrected_matrices_pqtls/"
    raw_matrices_save_path_rna = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/rna_descriptive/"
    how = "q"
    ##needed to update the covariates
    ##only load the status table once and pass it around to save on memory
    min_subject_threshold = 500
    most_frequent_val_max_freq = 0.95
    redo_collect_correct_pqtls = False
    redo_association_tests_prs = False
    redo_association_tests_pqtl = False
    redo_prs_pqtl_associations = False
    do_association_tests_rnaseq = True
    direction = False ##p values or direction from the assoc test. Set to False for P values
    correct_beforehand = False  ##keep off to use the model with built in correction for age, gender, and PCS
    redo_loader_saving = False
    if redo_collect_correct_pqtls:
        scores = combine_scores()
        if correct_beforehand:
            scores = correct_all_covariates("", use_precomputed_loader=True, precomputed_loader=scores)
        scores.to_csv(corrected_qtl_fname)
    if redo_loader_saving:
        plink_data_loaded = read_plink_bins_10K_index()
        correct_all_loaders(loaders=loaders_list, correct_beforehand=correct_beforehand, plink_data=plink_data_loaded,
                            min_subject_threshold=min_subject_threshold,
                            most_frequent_val_max_freq=most_frequent_val_max_freq)
    ##then, treat the PQTLS as the PRSES and test with all the dataloaders
    if redo_association_tests_prs:
        plink_data_loaded = None  ##to avoid pickling this when we send things to the queue
        update_covariates(
            status_table=status_table)  ##We use the 10 PCS from the GWAS and age and gender as covariates in the model, so let's keep them current
        make_test_all_loaders(loaders=loaders_list, which="PRSES", loop=how == "loop", direction = direction)
    if do_association_tests_rnaseq:
        genders = [0, 1, 2] #0 for women, 1 for men, and 2 for the combined cohort
        plink_data_loaded = None  ##to avoid pickling this when we send things to the queue
        update_covariates(status_table=status_table)
        make_test_all_loaders(loaders=loaders_list, which="RNA", loop=how == "loop", direction = direction, genders = genders, Y_linked_genes = Y_linked_genes)
    if redo_association_tests_pqtl:
        make_test_all_loaders(loaders=loaders_list, which="PQTLS", loop=how == "loop", direction = direction)
    if redo_prs_pqtl_associations:
        ##last, treat the PQTLS as dataloaders and tests with PRSES
        PQTLS_PRS_matrix = q_generate_prs_matrix(test="corrected_regression", duplicate_rows="last",
                                                 saveName=corrected_qtl_savename, tailsTest=None,
                                                 random_shuffle_prsLoader=False, use_prsLoader=True, direction = direction)
        PQTLS_PRS_matrix.to_csv("/net/mraid20/export/jasmine/zach/prs_associations/prs_pqtl_matrix.csv")
    make_figures = False
    if make_figures:
        s = stack_matrices_and_bonferonni_correct(fillwithNA=False)
        make_clustermaps(s)

