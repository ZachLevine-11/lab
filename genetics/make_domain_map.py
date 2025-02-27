import pandas as pd
import os
from os.path import isfile, join
from run_gwas import read_loader_in, loaders_list

def make_feature_maps(loaders_list, all_results_files):
    for loader in loaders_list:
        combo = pd.Series(list(set(list(read_loader_in(loader).columns)).intersection(set(all_results_files))))
        justname = str(loader).split(".")[2] + ".csv"
        combo.to_csv("/home/zacharyl/gwasInterface/inst/lists/" + justname)


if __name__ == "__main__":
    results_dir =  "/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_results/"
    all_results_files = list(map(lambda results_fname: results_fname.split("batch0.")[-1].split(".glm.linear")[0], [f for f in os.listdir(results_dir) if isfile(join(results_dir, f)) and f.endswith(".glm.linear") and f.startswith("batch0.")]))
    loaders_list = loaders_list[0:6]
    make_feature_maps(loaders_list, all_results_files)