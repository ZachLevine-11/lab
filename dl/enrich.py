import numpy as np
import pandas as pd
import gseapy as gp
from gseapy import barplot
from general_pred import read_res_file, multi_stage_correction, make_RNASeq_df
from matplotlib import pyplot as plt
from LabData.DataLoaders.SubjectLoader import SubjectLoader

gene_sets = ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "GO_Biological_Process_2023"]

def overrep(res):
    res = gp.enrichr(gene_list=list(res.loc[res["P*"] < 0.05, :].index.values),
                     gene_sets=gene_sets,
                     organism="Human",
                     outdir="/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/saved_predictors/gseapy").results.sort_values(by="Overlap", ascending=False)
    res["Term"] = res["Term"].str.split(" \(GO").str[0]
    barplot(res,
            column="Adjusted P-value",
            size=20,
            top_term=20,
            color="blue",
            title="ECG: Predicted Genes in Men")
    plt.title("Pathway Analysis: Predicted genes in men are implicated in the ETC")
    plt.savefig("/home/zacharyl/Desktop/myplot.png", dpi = 400)
    res = res.loc[res["Adjusted P-value"] < 0.05, :].sort_values("Adjusted P-value")
    return res

def do_ea(rna, gender, res):
    res = res.sort_values("P*", ascending = True)
    subs = SubjectLoader().get_data(study_ids=  list(range(100)) + list(range(1000, 1011, 1))).df.reset_index(["Date"], drop=True)
    subs = subs.loc[~subs.index.duplicated(keep="first"), "gender"]
    subs = subs.dropna().astype(int)
    this_gender = set(subs.loc[subs == gender].index.values)
    distance = pd.read_csv("~/Desktop/distance_matched.csv").set_index("RegistrationCode")
    distance = distance.loc[list(set(distance.index.intersection(this_gender))),:]
    distance_lower = distance.loc[distance["0"] < distance["0"].mean()]
    distance_upper = distance.loc[distance["0"] > distance["0"].mean()]
    distance_lower["dist"] = 0
    distance_upper["dist"] = 1
    distance = pd.concat([distance_lower, distance_upper])
    rna_merged = rna.merge(distance, left_index=True, right_index=True)
    dist = list(rna_merged["dist"].values)
    rna_merged = rna_merged.loc[:, res.index.values]
    rna_merged_transpose = rna_merged.iloc[:, 0:-3].T
    res = gp.gsea(data=rna_merged_transpose,
                  gene_sets= gene_sets,
                  cls=dist,
                  permutation_num=5000,
                  permutation_type='phenotype',
                  outdir=None,
                  method='log2_ratio_of_classes',
                  threads=96)
    print(res.res2d.loc[res.res2d["FWER p-val"] < 0.05,:].Term.values)
    return res

if __name__ == "__main__":
    women_emb = read_res_file("embeddings/RNA_0.csv")
    men_emb = read_res_file("embeddings/RNA_1.csv")
    women_emb = women_emb.loc[women_emb.index.get_level_values(1) == 1, :]
    men_emb = men_emb.loc[men_emb.index.get_level_values(1) == 1, :]
    women_emb = multi_stage_correction(women_emb.loc[women_emb.index.get_level_values(2) == 0.0, :]).reset_index(
        ["alpha", "weight"], drop=True)
    men_emb = multi_stage_correction(men_emb.loc[men_emb.index.get_level_values(2) == 0.0, :]).reset_index(
        ["alpha", "weight"], drop=True)
    #rna_df = make_RNASeq_df()
    #men = do_ea(rna_df, 1,  men_emb)
    #men.res2d.sort_values("FDR q-val")[["Term", "FDR q-val"]]
