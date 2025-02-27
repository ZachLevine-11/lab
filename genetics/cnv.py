import numpy as np
import pandas as pd
import os
from os.path import isfile, join
from GeneticsPipeline.config import qp_running_dir
from GeneticsPipeline.helpers_genetic import read_status_table
from statsmodels.formula.api import ols
from LabQueue.qp import qp
from LabData.DataLoaders.CGMLoader import CGMLoader
from run_gwas import read_loader_in
from LabUtils.addloglevels import sethandlers
from sklearn.decomposition import PCA

def make_filemap(basepath, status_table_qc):
    status_table_qc = status_table_qc.set_index("gencove_id")
    all_gencove_ids_passed_qc = set(status_table_qc.index.values)
    alldirs = os.listdir(basepath)
    filemap = {}
    for dir in alldirs:
        cnsfile, cnrfile = None, None
        subdirs = os.listdir(join(basepath, dir))
        ##only subdirs that passed qc
        gencove_ids_passed_qc = list(set.intersection(set(subdirs), all_gencove_ids_passed_qc))
        if len(gencove_ids_passed_qc) > 1:
            gencove_ids_passed_qc = gencove_ids_passed_qc[-1] ##sort in ascending order by date, so that the last entry is the latest one. For people with multiple valid samples, pick the last one
        if len(gencove_ids_passed_qc) > 0:
            tenkCode = status_table_qc.loc[gencove_ids_passed_qc, "RegistrationCode"].values[0]
            containedfiles = os.listdir(join(basepath, dir, gencove_ids_passed_qc[0]))
            if gencove_ids_passed_qc[0] + "_cnv-cns.cns" in containedfiles:
                cnsfile = join(basepath, dir, gencove_ids_passed_qc[0]) + "/" + gencove_ids_passed_qc[
                    0] + "_cnv-cns.cns"
            else:
                pass
            if gencove_ids_passed_qc[0] + "_cnv-cnr.cnr" in containedfiles:
                cnrfile = join(basepath, dir, gencove_ids_passed_qc[0]) + "/" + gencove_ids_passed_qc[
                    0] + "_cnv-cnr.cnr"
            else:
                pass
            filemap[tenkCode] = {"cns_file_loc": cnsfile,
                                 "cnr_file_loc": cnrfile}
    filemap = pd.DataFrame(filemap).T
    filemap.index.name = "RegistrationCode"
    filemap.to_csv("/net/mraid20/export/jasmine/zach/cnv/filemap.csv")
    return filemap

##for each bin, make a df for the assoc test with log2 depth
def make_bindf(bin_number, col, filemap_loc = "/net/mraid20/export/jasmine/zach/cnv/filemap.csv"):
    filemap = pd.read_csv(filemap_loc).set_index("RegistrationCode").dropna(subset = [col])
    weight_id = {"cns_file_loc": 7,
                 "cnr_file_loc": 6}
    bindf = {}
    for person in filemap.index.values:
        person_bin = list(*pd.read_csv(filemap.loc[person, col], sep="\t", nrows=1, skiprows=bin_number).values)
        id = str(person_bin[0]) + "_" + str(person_bin[1]) + "_" + str(person_bin[2])
        log2 = person_bin[4]
        depth = person_bin[5]
        weight = person_bin[weight_id[col]]
        bindf[person] = {"id": id,
                         "log2": log2,
                         "depth": depth,
                         "weight": weight}
    bindf = pd.DataFrame(bindf).T
    bindf.index.name = "RegistrationCode"
    bindf_subdir = {"cns_file_loc": "/net/mraid20/export/jasmine/zach/cnv/bindfs_cns/",
                    "cnr_file_loc": "/net/mraid20/export/jasmine/zach/cnv/bindfs_cnr/"}
    bindf.to_csv(bindf_subdir[col] + str(bin_number) + ".csv")
    return bindf

def assoc_test_individual_bin(bin_number, col, filemap_loc = "/net/mraid20/export/jasmine/zach/cnv/filemap.csv"):
    status_table_all = read_status_table()
    status_table_all = status_table_all.sort_values(by=["date", "version"], ascending=True)
    try:
        status_table_qc = status_table_all[status_table_all.passed_qc].copy()
    except ValueError:
        status_table_qc = status_table_all.dropna()
        status_table_qc = status_table_qc[status_table_qc.passed_qc].copy()
    bindf_subdir = {"cns_file_loc": "/net/mraid20/export/jasmine/zach/cnv/bindfs_cns/",
                    "cnr_file_loc": "/net/mraid20/export/jasmine/zach/cnv/bindfs_cnr/"}
    cache_files = os.listdir(bindf_subdir[col])
    bin_cache_exists = str(bin_number) + ".csv" in cache_files
    if not bin_cache_exists:
        bindf = make_bindf(bin_number, col, filemap_loc = filemap_loc)
    else:
        print("found cache for bin: " + str(bin_number))
        bindf = pd.read_csv(bindf_subdir[col] + str(bin_number) + ".csv").set_index("RegistrationCode")
#    assert len(set(bindf.id)) == 1 ##otherwise something went wrong
    covars = pd.read_csv("/net/mraid20/export/jasmine/zach/cnv/pcs.csv").set_index("RegistrationCode")
    pheno = read_loader_in(CGMLoader)["ea1c"]
    bindf = bindf["log2"]
    bindf.name = "coverage"
    bindf.index.name = "RegistrationCode"
    fancydf = pd.merge(bindf, pheno, left_index=True, right_index=True,how="inner").dropna(axis=0).merge(covars, left_index=True,right_index=True,how="inner")
    fancydf = fancydf[["coverage", pheno.name,
                       "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
                       "PC11", "PC12", "PC13", "PC14", "PC15", "PC16", "PC17", "PC18", "PC19", "PC20",
                       "age","gender"]]
    fancydf["y"] = fancydf[pheno.name]
    ##equilvalent to --variance standardize, zero mean and unit variance everything we model
    fancydf = (fancydf - fancydf.mean()) / fancydf.std(ddof=0)  ##dd0f is default 1 in pandas and 0 in numpy and in plink, allign them
    ##if the coverage column is not numeric, then we use the numbers as levels instead of the numbers we want
    fancydf["coverage"] = fancydf["coverage"].astype(float)
    formula = "y ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 +  PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20 + gender + age + coverage"
    try:
        model = ols(formula, fancydf).fit()
        hypotheses = "(coverage = 0)"
        test_ = model.wald_test(hypotheses)
        pval = float(test_.pvalue)
        return pval
    except Exception as e:
        return None

##make the set of reliable bins
def make_binset(thresh, filemap, col = "cnr_file_loc"):
    template = pd.read_csv(filemap[col][0],
                           sep="\t")  ##the reliability of each bin is stored in each person's individual cnv data, it's the same across all people so just grab it from the first person
    bins = list(
        template.loc[template.weight > template.weight.mean() + thresh * template.weight.std(ddof=0), :].index.values)
    return bins

def do_assoc_tests(filemap, col = "cnr_file_loc", filemap_loc = "/net/mraid20/export/jasmine/zach/cnv/filemap.csv", res_fname = "/net/mraid20/export/jasmine/zach/cnv/ea1c_test.csv", thresh = 1):
    filemap_col  = filemap[col].dropna()
    template = pd.read_csv(filemap_col[0], sep = "\t")
    res = {}
    with qp(jobname="cnv", delay_batch=30) as q:
        q.startpermanentrun()
        bins = make_binset(thresh, filemap, col)
        for bin_number in range(len(bins)):
            res[bin_number] = q.method(assoc_test_individual_bin, (bin_number, col, filemap_loc))
        for k, v in res.items():
            try:
                res[k] = q.waitforresult(v)
            except Exception:
                res[k] = None
    for k, v in res.copy().items(): ##because we're deleting keys as we iterate, make sure we iterate over a copy of the original dictionary
        if v is None:
            del res[k]
    res = pd.Series(res)
    res.to_csv(res_fname)
    return res

def format_cnv_results(filemap, col = "cnr_file_loc", results_fname = "/net/mraid20/export/jasmine/zach/cnv/ea1c_test.csv", thresh =1):
    filemap_col  = filemap[col].dropna()
    template = pd.read_csv(filemap_col.iloc[0], sep = "\t")
    res = pd.DataFrame(dict(zip(range(len(template)), template.apply(lambda singleRow: {"CHR": singleRow[0], "START": singleRow[1], "END": singleRow[2]}, axis = 1)))).T
    ##the ingeger index of the fname is meaningless, but the first column holds the bin number corresponding to each p value
    mapper = pd.read_csv(results_fname).set_index("Unnamed: 0").to_dict()["0"]
    res["P"] = list(map(mapper.get, res.index.values))
    ##remove unreliable bins
    res = res.iloc[make_binset(thresh, filemap, col),:]
    res.to_csv(results_fname.split(".csv")[0] + "_formatted.csv")
    return res

def make_geno(filemap, col = "cnr_file_loc", thresh = 0.4):
    template = pd.read_csv(filemap.loc[filemap.index.values[0], col], sep="\t")
    template["id"] = list(map(lambda x: str(x[0]) + "_" + str(x[1]) + "_" + str(x[2]),
                             zip(template["chromosome"], template["start"], template["end"])))
    geno = pd.DataFrame(np.zeros([len(filemap), len(template)]))
    geno.index = filemap.index
    geno.columns = template["id"]
    geno = geno.T
    bins = make_binset(thresh, filemap, col)
    i = 0
    for person in geno.T.index:
        print(i)
        try:
            read_in = pd.read_csv(filemap.loc[person, col], sep = "\t")
            read_in = read_in.iloc[bins,:] ##filter out unreliable bins before PCS
            read_in["id"] = list(map(lambda x: str(x[0]) + "_" + str(x[1]) + "_" + str(x[2]), zip(read_in["chromosome"], read_in["start"],  read_in["end"])))
            person_cols = read_in[["id", "log2"]].set_index("id")
            geno[person] = person_cols
        except Exception:
            pass
        i += 1
    geno = geno.T
    return geno

def do_pca(geno, status_table_qc, wd = "/net/mraid20/export/jasmine/zach/cnv/", fname = "pcs.csv", howmany = 20):
    pca = PCA()
    pca.fit(geno)
    pcs = pd.DataFrame(pca.components)
    pcs["RegistrationCode"] = geno.index
    pcs = pcs.set_index("RegistrationCode")
    age_gender_df = pd.read_csv("/net/mraid20/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt", sep="\t")
    age_gender_df["RegistrationCode"] = age_gender_df["IID"].apply(status_table_qc.set_index("gencove_id").RegistrationCode.to_dict().get)
    age_gender_df = age_gender_df.drop("IID", axis=1).set_index("RegistrationCode", drop=True)[["gender", "age"]]
    subset_pcs = pcs.iloc[:, 0:howmany]
    subset_pcs = subset_pcs.merge(age_gender_df, how = "inner", left_index = True, right_index = True)
    subset_pcs.columns = list(map(lambda x: x[0] + str(x[1]), zip(["PC" for x in range(howmany + 1)], list(range(1, howmany + 1, 1))))) + ["gender" , "age"]
    subset_pcs.to_csv(wd + fname)

##the col parameter selects if we want single bins or segments of bins, i.e which level of CNV reolution we want
if __name__ == "__main__":
    os.chdir(qp_running_dir)
    sethandlers()
    status_table_all = read_status_table()
    status_table_all = status_table_all.sort_values(by=["date", "version"], ascending=True)
    try:
        status_table_qc = status_table_all[status_table_all.passed_qc].copy()
    except ValueError:
        status_table_qc = status_table_all.dropna()
        status_table_qc = status_table_qc[status_table_qc.passed_qc].copy()
    basepath = "/net/mraid20/export/genie/10K/genetics/Gencove/RawData/"
    use_cache_filemap = True
    if not use_cache_filemap:
        filemap = make_filemap(basepath, status_table_qc)
    else:
        filemap = pd.read_csv("/net/mraid20/export/jasmine/zach/cnv/filemap.csv").set_index("RegistrationCode")
    do_assoc = True
    if do_assoc:
    ##note that the cnr binaries have all windows in the same order
    ##this is not true of the cns binaries, so different window numbers correspond to different windows
    ##only use cnr files for now. the windows in cns files are not universal.
        assoc = do_assoc_tests(filemap, col = "cnr_file_loc")
