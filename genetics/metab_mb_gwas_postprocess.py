import numpy as np
import pandas as pd
from run_gwas import summarize_gwas
from dl_utils import rsid_to_gene
import gseapy as gp
from statsmodels.stats.multitest import multipletests

from LabData.DataAnalyses.MBSNPs.taxonomy import taxonomy_df

def mb_examine():
    g, rank = summarize_gwas(numGwases = 400,
        clump_dir='/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/microbiome/gwas_results_clumped/')
    tax = taxonomy_df().set_index("SGB")
    repNumbers = ["Rep_" + k.split("__")[-1].split(".clumped")[0] for k,v in g.items()]
    return tax.loc[repNumbers, :]

##for each metabolite take its significant snps, map to genes, and then run pathway analysis on the genes
def pathways_metab(gwas_results, rank):
    res = {}
    i = 0
    for k,v in gwas_results.items():
        print("Onto metabolite :", i,  "/" , len(rank))
        metabName =  k.split("__")[-1].split(".clumped")[0]
        metabSnps = list(v.loc[v.P < (5*10**(-8))/3107,:].ID)
        metabGenes = list(map(rsid_to_gene, metabSnps))
        ##remove Nones and the read in cached Nones which get read as Nans and need their own check
        metabGenes = list(filter(lambda item: item != [None] and type(item[0]) != float, metabGenes))
        ##flatten thelist
        metabGenes = [x for y in metabGenes for x in y]
        if metabGenes != []:
            try:
                res[metabName] = gp.enrichr(gene_list = metabGenes,
                                        gene_sets = ["GO_Biological_Process_2021"],
                                        organism = "Human",
                                        outdir = "/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/metab/enrichment",
                                        cutoff = 0.5).results.sort_values(by = "Overlap", ascending = False)
            except Exception:
                res[metabName] = None
        else:
            res[metabName] = None
        i += 1
    for k, v in res.items():
        if v is not None:
            res[k]["Metabolite"] = k
    res_df =  pd.concat(res.values())
    res_df.to_csv("/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/metab/enrichment/Metab_Enrichment.csv")
    return res_df

##GSEApy adjusts each metabolite independently for the pathways tested using bh_fdr in statsmodels api
##If we take single metabolite's original P values and correct them this way, we get the same result as the adjusted p values the software provides
# ## metab["Multiple-Test-Adjusted-P-Value"] = multipletests(pvals=metab["P-value"].to_numpy().flatten(), method="fdr_bh")[1]
##For consistency with gwas, use bonferonni
##adjust for all metabolites and all pathways tested
def read_correct_pathways(multi_trait_correction = False, drop_duplicates =True):
    res  = pd.read_csv("/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/metab/enrichment/Metab_Enrichment.csv").drop("Unnamed: 0", axis =1)
    if multi_trait_correction:
        res["Multiple-Test-Adjusted-P-Value"] =multipletests(pvals=res["P-value"].to_numpy().flatten(), method="bonferroni")[1]
        res = res.loc[res["Multiple-Test-Adjusted-P-Value"] < 0.05, :]
        res = res.sort_values(by = "Multiple-Test-Adjusted-P-Value")
    else: ##use original columns
        res = res.loc[res["Adjusted P-value"] < 0.05, :]
        res = res.sort_values(by="Adjusted P-value")
    res["Go"] = list(map(lambda str: str.split("(")[-1].split(")")[0], res["Term"]))
    res["Pathway"] = list(map(lambda str: str.split(" (")[0], res["Term"]))
    res["Metabolite"] = list(map(lambda str: str.split("batch0.")[-1], res["Metabolite"]))
    if drop_duplicates:
        res = res.drop_duplicates(subset = ["Pathway"], keep = "first") ##since we sorted by ascending order, the smallest association for each repeated pathway will be first, so grab it
    return  res

def read_noam_annotations():
    annot = pd.read_csv("/net/mraid20/export/jafar/Microbiome/Analyses/Noamba/pheno/lipidomics/annotated_lipids.csv")
    return annot

def merge_noam_annotations_res():
    res = read_correct_pathways(multi_trait_correction=False)
    annot = read_noam_annotations()
    merged = pd.merge(res, annot, left_on = "Metabolite", right_on = "name", how = "inner")
    return merged

if __name__ == "__main__":
    redo_pathways = True
    if redo_pathways:
        g, rank = summarize_gwas(numGwases=3645,
                             clump_dir='/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/metab/gwas_results_clumped/')
        res = pathways_metab(g, rank)