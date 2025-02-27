from LabData.DataLoaders.PRSLoader import PRSLoader
import numpy as np
import pandas as pd
from cluster_things_old import print_clusters


def generate_ensembles(fname="~/Desktop/120_prs_250_sim.csv", do_corr=True, threshold=0.8):
    clusters_names = {}
    clust_col_identity = print_clusters(df=pd.read_csv(fname).set_index("Unnamed: 0"), do_corr=do_corr, noPrint=True,
                                        threshold=threshold)
    longest_cluster_size = 1
    for identity in range(clust_col_identity.max()):
        cluster = clust_col_identity[clust_col_identity.eq(identity)]
        if len(cluster) > 1:  ##only grab clusters with size   > 1
            if len(cluster) > longest_cluster_size:
                longest_cluster_size = len(cluster)
            for thename in cluster.index:
                clusters_names[thename] = cluster[thename]
    theprs = PRSLoader().get_data().df
    descriptionmap = PRSLoader().get_data().df_columns_metadata  ##assumes that the order of the metadata is the same
    # print(sum(theprs.columns == descriptionmap.index)) ##the index is the same, so this works
    ranges = list()
    for cluster in range(clust_col_identity.max()):
        ranges.append(list())
    descriptionmap.h2_description = list(map(lambda colname: str(colname)[0:30],
                                             descriptionmap.h2_description.values))  ##make the names in h2_description shorter
    descriptionmap.index = descriptionmap.h2_description  ##make this the index so we can search it
    for prs in clusters_names.keys():
        prs_column_index = descriptionmap.index.get_loc(prs)
        if isinstance(prs_column_index, int):  ##arrays of Falses indicate we couldn't find that prs
            ranges[clusters_names[prs]].append(prs_column_index)
    ranges = np.array(ranges)
    ensembles = []
    for i in range(len(ranges)):
        if len(ranges[i]) > 0:
            ranges[i] = ranges[i]
            ensembles.append(np.mean(theprs.iloc[:, ranges[i]], 1))
    ensemble_PRSes = pd.concat(ensembles, axis=1)
    return ensemble_PRSes
