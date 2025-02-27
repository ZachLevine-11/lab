import numpy as np
import pandas as pd
from LabData.DataLoaders.PRSLoader import PRSLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader  ##for tests
from LabData.DataLoaders.SerumMetabolomicsLoader import data_cache_10k_RT_clustering_pearson08
from modified_tom_functions import correct_all_covariates


##preprocessing for PRSes
##prs_from_loader corresponds to a column in PRSLoader.get_data().df
##data arguments
def preprocess_loader_data(keep_prs=True, duplicate_rows="last", prs_from_loader=None, saveName=None,
                           random_shuffle_prsLoader=False,
                           use_prsLoader=True):  ##we can either average values for duplicate rows, or just keep the last value for each
    data = pd.read_csv("/net/mraid20/export/jasmine/zach/prs_associations/corrected_loaders/" + saveName)
    prses = pd.read_pickle(
        "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/rna_options_final/" + "without_batch_correction_only_filtered_10000_no_regress_5mln_sample.df")
    prses = np.log1p(np.clip(prses, -0.999, None, ))
    loaded_prs = prses[prs_from_loader]  # load prs with loader, assume that prs_from_loader is not none
    if random_shuffle_prsLoader:
        ##shuffle all columns
        ##the line bwlow shuffles every column, don't do it
        # loaded_prs = loaded_prs.apply(np.random.permutation)
        ##we need to shuffle the index when its still the index, making it the column of the data frame doesn't work to shuffle for some reason
        loaded_prs.index = loaded_prs.index[np.random.permutation(len(loaded_prs))]
    loaded_prs = loaded_prs.reset_index()  ##do this after we scramble the index. We can't shuffle index if it's a column for some reason.
    ##prs might be duplicated, if it is, keep only one
    loaded_prs = loaded_prs.loc[:, ~loaded_prs.columns.duplicated()]
    trimsd = loaded_prs[(loaded_prs[prs_from_loader] < loaded_prs[prs_from_loader].quantile(0.95)) & (
                loaded_prs[prs_from_loader] > loaded_prs[prs_from_loader].quantile(0.05))][prs_from_loader].std()
    loaded_prs[prs_from_loader] = (loaded_prs[prs_from_loader] - loaded_prs[prs_from_loader].mean()) / trimsd
    ##should normalize the prs and grab other relevant info
    if keep_prs:
        ##the name of the prs right now is that prs - rename just to prs
        loaded_prs = loaded_prs.rename({prs_from_loader: "prs"}, axis=1)
    else:
        lower_bound = loaded_prs[prs_from_loader].mean() + 2 * trimsd
        upper_bound = loaded_prs[prs_from_loader].mean() - 2 * trimsd
        loaded_prs.loc[:, ["PRS_class"]] = 0  ##0 if neither high nor low PRS
        loaded_prs.loc[loaded_prs[prs_from_loader] >= lower_bound, "PRS_class"] = 2  # 2 for high prs
        loaded_prs.loc[loaded_prs[prs_from_loader] <= upper_bound, "PRS_class"] = 1  # 1 for low prs
        loaded_prs = loaded_prs.drop(prs_from_loader,
                                     1)  ##If we're dropping the prs column, the name of it doesn't matter
    prs = loaded_prs
    # The first 10 digits of metabolomics registration codes, prefixed by "10_K" correspnd to 10K registration codes
    dataprs = pd.merge(prs, data, left_on="RegistrationCode", right_on="RegistrationCode", how="inner")
    if duplicate_rows == "mean":
        dataprs = dataprs.groupby('RegistrationCode').mean()
    elif duplicate_rows == "last":
        dataprs = dataprs.drop_duplicates(subset='RegistrationCode', keep="last")
    return dataprs  ##RegistrationCodes is a column, not an index


def test_random_shuffle():
    dataprs = preprocess_loader_data(loader=BodyMeasuresLoader, use_clustering=True, duplicate_rows="last",
                                     keep_prs=False, index_is_10k=True, usePath=False, prs_path="",
                                     prs_from_loader="102_irnt", use_imputed=True,
                                     use_corrected=True, saveName="body_corrected.csv", get_data_args=None,
                                     random_shuffle_prsLoader=True)
