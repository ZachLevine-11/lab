import json
import ast
import os
import numpy as np
import pandas as pd
import os
from os.path import isfile, join
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader ##keep here
from GeneticsPipeline.config import qp_running_dir
from functools import reduce

def snpDbExtractor(thestr):
    genes = [None]
    if "in the" in thestr:
        genes = [thestr.split("in the")[-1].split("gene")[0].split("+")[-1].split("&")[0]]
    elif "near genes" in thestr:
        containing_str = thestr.split("near genes")[-1].split("chromosome")[0].split(":")
        onlyGenes = [x for x in containing_str if "prev-item" in x]
        genes = [x.split("&")[0] for x in onlyGenes]
    return genes

##return a list in case there are multiple genes we are near instead of one gene the snp is in
def rsid_to_gene(rsid):
    cached_snps = [f for f in os.listdir("/net/mraid20/export/jasmine/zach/height_gwas/snp_cache") if isfile(join("/net/mraid20/export/jasmine/zach/height_gwas/snp_cache", f))]
    if rsid + '.csv' in cached_snps:
        genes = list(pd.read_csv("/net/mraid20/export/jasmine/zach/height_gwas/snp_cache/" + rsid + ".csv")["0"])
        return genes
    import urllib.request ##Needs to be called explicitly to have access to this module
    baseUrl = "https://www.ncbi.nlm.nih.gov/search/all/?term="
    theUrl = baseUrl + rsid
    response = urllib.request.urlopen(theUrl)
    content = response.read().decode("UTF-8")
    genes = snpDbExtractor(content)
    pd.Series(genes).to_csv("/net/mraid20/export/jasmine/zach/height_gwas/snp_cache/" + rsid + ".csv")
    return genes

def rsid_to_gene_batched(batch):
    return pd.Series({rsid: rsid_to_gene(rsid) for rsid in batch})

##Only works in debug mode with the queue for some reason
##very computationally constly, don't run
def annotate_all_snps(batch_width = 10):
    os.chdir("/net/mraid20/export/mb/logs/")
    with qp(jobname="snp", delay_batch=30) as q:
        q.startpermanentrun()
        binaries = read_plink1_bin(
            bed="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc.bed",
            bim="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc.bim",
            fam="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc.fam")
        snplist = binaries.snp.to_series().reset_index()["snp"]
        numSnps = len(snplist)
        snps_batched = np.array_split(snplist, numSnps//batch_width)
        snpMap = {}
        print("Starting to map genes for ", numSnps, " SNPs")
        i = 0
        del snplist, binaries
        for batch in snps_batched:
            snpMap[i] = q.method(rsid_to_gene_batched, (batch,))
            i += 1
        snpMap = {k: q.waitforresult(v) for k, v in snpMap.items()}
    snpMap = pd.concat(snpMap.values(), axis=0)  ##in order of columns
    snpMap.to_csv("/net/mraid20/export/jasmine/zach/dl/snpGeneMap.csv")
    return snpMap

##For each gene, obtan a list of snps that are in or near to it
##inverse of rsid_to_gene
##Requires annotation of all SNPS, which is computationally costly
def inverse_snp_map(snpMap_fname = "/net/mraid20/export/jasmine/zach/dl/snpGeneMap.csv"):
    snpMap = pd.read_csv(snpMap_fname).dropna()
    snpMap.columns = ["rsid", "genes"]
    geneMap = {}
    for snp in snpMap["rsid"]:
        if type(snpMap.loc[snpMap["rsid"] == snp, "genes"]) != float:
            geneSet = snpMap.loc[snpMap["rsid"] == snp, "genes"].values[0].strip("[]").split(",")
            for gene in geneSet:
                formatted_gene = gene.strip("' ""  ")
                if geneMap.get(formatted_gene) is None:
                    geneMap[formatted_gene] = [snp]
                else:
                    geneMap[formatted_gene].append(snp)
    return pd.Series(geneMap)

##From https://medium.com/intothegenomics/annotate-genes-and-genomic-coordinates-using-python-9259efa6ffc2
def read_gene_annotation_file(fname = "/net/mraid20/export/jasmine/zach/dl/gene_annotation.gtf"):
    gencode = pd.read_table(fname, comment = "#", sep = "\t", names = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand',
                                                   'frame', 'attribute'])
    gencode["gene"] = list(map(lambda x: x.split("gene_name ")[-1].split(";")[0].strip('"'), gencode["attribute"]))
    gencode = gencode.drop(["source", "score"], axis = 1)
    return gencode

def gene_name_to_position(gene, gencode):
    onlyGene = gencode.loc[gencode.gene == gene,:]
    onlyGene = onlyGene.loc[onlyGene["feature"] == "gene"]
    ans  = None
    if len(onlyGene) == 1: ##catch genes with no match and with more than one match
        ##the only genes with more than one match are on the sex chromosomes, which we don't have data for anyway
        ##so exclude them
        ##keep chr as a string to match plink binaries
        ans = {"start": int(onlyGene["start"]),
               "end": int(onlyGene["end"]),
               "chr": str(onlyGene["seqname"]).split("chr")[-1].split("\n")[0]}
    return ans

def gene_to_rsids(gene, plink_bins, gencode):
    pos = gene_name_to_position(gene, gencode)
    if pos is None:
        return pos
    bins_chr = plink_bins.loc[:,plink_bins["chrom"] == pos["chr"]]
    bins_chr_after_start = bins_chr.loc[:, bins_chr["pos"] >= pos["start"]]
    bins_chr_after_start_before_stop = bins_chr_after_start.loc[:, bins_chr_after_start["pos"] <= pos["end"]]
    snps_inside = bins_chr_after_start_before_stop.snp.to_pandas().reset_index(drop = True)
    return list(snps_inside)

def loadGeneSets(file = "/net/mraid20/export/jasmine/zach/dl/h.all.v2022.1.Hs.json"): ##open needs the full file path, i.e without the "~"
    with open(file) as f:
        data = json.load(f)
    concise_gene_sets = {}
    for k, v in data.items():
        concise_gene_sets[k] = v["geneSymbols"]
    return concise_gene_sets

##for each gene set based on a pathway, find all the SNPS in those genes.
def makeSNPSets(file = "/net/mraid20/export/jasmine/zach/dl/h.all.v2022.1.Hs.json", cached = True, snp_cache_fname  = "/net/mraid20/export/jasmine/zach/dl/SNPSetCache.csv"):
    if cached:
        cached_snps = pd.read_csv(snp_cache_fname).rename({"Unnamed: 0" :"gene"}, axis = 1).set_index("gene")
        ##the SNPSets are read in as strings from the cache so use eval to make them lists before we iterate over them
        snpsets_concise =  {col: list(filter(lambda snp: snp != ".", [item for sublist in cached_snps[col].dropna() for item in ast.literal_eval(sublist) if sublist is not None])) for col in cached_snps.columns}
        return snpsets_concise
    else:
        gencode = read_gene_annotation_file()
        plink_bins = read_plink1_bin(
                bed="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bed",
                bim="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bim",
                fam="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.fam")
        genes = loadGeneSets(file)
        snps = {}
        numGenes = len(genes)
        i = 0
        for k, v in genes.items():
            print("Now onto gene set " + str(i) + "/" + str(numGenes))
            snps[k] = {gene: gene_to_rsids(gene, plink_bins, gencode) for gene in v}
            i += 1
        snps = pd.DataFrame(snps)
        snps.index.name = "gene"
        snps.to_csv(snp_cache_fname)
        snpsets_concise = {col: list(filter(lambda snp: snp != ".", [item for sublist in snps[col].dropna() for item in sublist if sublist is not None]))
                            for col in snps.columns}
        return snpsets_concise

##build the following data for a SNPSet: [numPeople, genotypes], where genotypes is the genotype for each SNP
def extract_snps_from_bins(SNPSet, cache_dir, i):
    try:
        ##load in here as opposed to sending each q job with the full binaries already
        plink_bins = read_plink1_bin(
                bed="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bed",
                bim="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bim",
                fam="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.fam")
        ##exclude "." SNPs missing an rsid, otherwise they match to more than one location in the binaries
        SNPSet = list(filter(lambda snp: snp != ".", SNPSet))
        SNPArray = np.empty([len(plink_bins), len(SNPSet)])
        for j in range(len(SNPSet)):
            SNPArray[:, j] = plink_bins.loc[:, plink_bins.snp==SNPSet[j]].values.flatten()
        np.save(file=cache_dir + str(i), arr=SNPArray)
        ##save memory by not returning the SNP array object and only saving it to disk
        return None
    except Exception:
        return None

##for each pathway save all the training data, which is of dimension [numPeople, numSnps], where numSnps is the number of SNPS in any gene in that gene set in the cache
##the forward method of the network pulls data directly from here
def make_all_train_data(cache_dir = "/net/mraid20/export/jasmine/zach/dl/cache/", cached_snpsets = False, gene_pathway_fname = "/net/mraid20/export/jasmine/zach/dl/h.all.v2022.1.Hs.json"):
    ##for each pathway, we don't care about genes anymore so collapse all the snps from different genes into one array
    ##map pathway: SNPS
    ##for each person, you have an array of [SNPSets, maxSNPSetLength]
    ##for each person for each SNPSet, you have an array of MaxSNPSetLength (padded by Nones), where each entry is 0,1,2 corresponding to the genotype at each SNP
    tickets = {}
    os.chdir(qp_running_dir)
    with qp(jobname="dl", delay_batch=30) as q:
        q.startpermanentrun()
        ##do after q.startpermanentrun to avoid this getting sent to the queue for no reason to do nothing
        snpSets = makeSNPSets(file= gene_pathway_fname, cached=cached_snpsets)
        i = 0
        existing_cache = os.listdir(cache_dir)
        for snpSet in snpSets.values():
            print("Onto SNPSet {}".format(i))
            if str(i) + ".npy" not in existing_cache:
                tickets[i] = q.method(extract_snps_from_bins, (snpSet, cache_dir, i))
            else:
                print("Found existing cache for SNPSet: " + str(i) + ", skipping.")
            i += 1
        for k, v in tickets.items():
            try:
                tickets[k] = q.waitforresult(v)
            except Exception:
                pass

##map index in genetics data first dimension to corresponding RegistrationCode
##the cached version works on mcluster11, the not cached version does not
def get_train_data_registration_codes(plink_bins = None, cached = True, cached_fname = "/net/mraid20/export/jasmine/zach/dl/id_map.csv"):
    if cached: ##so as to not have to pass the plink bins in when using mcluster02 that doesn't support reading them in
        return pd.read_csv(cached_fname)["0"].to_dict()
    status_table = read_status_table()
    gencove_codes_from_bins = plink_bins.sample.to_pandas().values
    tenkCodes_same_order = list(map(status_table.set_index("gencove_id").RegistrationCode.to_dict().get, gencove_codes_from_bins))
    theMap = pd.Series(tenkCodes_same_order)
    theMap.to_csv(cached_fname)
    theMap = theMap.to_dict()
    return theMap

##read in a phenotype and allign it to the genetics binaries
def create_Y():
    idMap = get_train_data_registration_codes(cached=True)
    idMap_Series = pd.Series(idMap)
    idMap_Series.name = "RegistrationCode"
    Y = BodyMeasuresLoader().get_data(study_ids="10K", groupby_reg="latest").df.reset_index().set_index("RegistrationCode")
    Y = pd.merge(left = Y, right = idMap_Series, left_index = True, right_on = "RegistrationCode").set_index("RegistrationCode")["height"]
    return Y

def do_single_person_series(gencove_id, i, cache_dir = "/net/mraid20/export/jasmine/zach/dl/person_cache/"):
    bins = read_plink1_bin(
        bed="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bed",
        bim="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bim",
        fam="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.fam")
    pd.Series(list(*bins.loc[bins.sample.values == gencove_id].values)).to_csv(cache_dir + str(i) + ".csv")

def make_efficient_cache(cache_dir = "/net/mraid20/export/jasmine/zach/dl/person_cache/"):
    bins = read_plink1_bin(
        bed="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bed",
        bim="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bim",
        fam="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.fam")
    people = list(bins.sample.values)
    del bins ##before q.startpermanentrun
    tickets = {}
    os.chdir(qp_running_dir)
    i = 0
    with qp(jobname="dl", delay_batch=30) as q:
        q.startpermanentrun()
        ##do after q.startpermanentrun to avoid this getting sent to the queue for no reason to do nothing
        existing_cache = os.listdir(cache_dir)
        for person in people:
            if str(i) + ".csv" not in existing_cache:
                tickets[person] = q.method(do_single_person_series, (person, i))
            else:
                pass
            i += 1
        for k, v in tickets.items():
            try:
                tickets[k] = q.waitforresult(v)
            except Exception:
                pass

def make_snplist(fname = "/net/mraid20/export/jasmine/zach/dl/snplist.csv"):
    bins = read_plink1_bin(
        bed="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bed",
        bim="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bim",
        fam="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.fam")
    pd.Series(bins.snp.values[1:len(bins.snp.values)]).to_csv(fname)

def mask_single_gene(gene):
    plink_bins = read_plink1_bin(
        bed="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bed",
        bim="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.bim",
        fam="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_final_pass.fam")
    gencode = read_gene_annotation_file()
    snplist = pd.read_csv("/net/mraid20/export/jasmine/zach/dl/snplist.csv")["0"]
    snps_contained = gene_to_rsids(gene, plink_bins, gencode)
    ##for each snp return a Series the same length as snp sequence from genetics binaries with -1 everhwere except the location of the SNP
    which_are_snps = reduce(lambda x, y: x.add(y, fill_value=0), list(map(snplist.str.find, snps_contained)))
    ##when we sum the list of these series and reduce to a single series we end up with -numSnps at locs with no SNP, and -(numSnps) + 1 where we have a snp in the gene
    which_are_snps = which_are_snps.mask(which_are_snps == -which_are_snps, 0)  ##set all useless positions to zero
    ##set all useful positions to 1
    which_are_snps = which_are_snps(which_are_snps != 0, 1)
    return which_are_snps

def create_snp_to_gene_mask(masks_fname = "/net/mraid20/export/jasmine/zach/dl/masks/mask0.npy"):
    genes = loadGeneSets()
    ##the mask gets multiplied by the weights matrix of the second layer in the network as weight*mask (matmul)
    ##ith column of the mask matrix corresponds to the ith gene in the snpSet
    ##we have 0 in the ith row of the mask matrix unless the corresponding SNP talks to that the ith gene, in which case one
    tickets = {}
    os.chdir(qp_running_dir)
    with qp(jobname="mask", delay_batch=30) as q:
        q.startpermanentrun()
        snplist = pd.read_csv("/net/mraid20/export/jasmine/zach/dl/snplist.csv")["0"]
        allgenes_unique = list(set.union(*list(map(lambda geneset: set(geneset), genes.values()))))
        numGenes = len(allgenes_unique)
        mask = np.zeros([numGenes, len(snplist)], dtype=int)
        for gene in allgenes_unique:
            tickets[gene] = q.method(mask_single_gene, (gene,))
        for k, v in tickets.items():
            try:
                tickets[k] = q.waitforresult(v)
            except Exception:
                pass
    i = 0
    for k,v in tickets.items():
        mask[i, :] = tickets[k]
        i += 1
    np.save(masks_fname, mask)
    return mask

def create_gene_to_pathway_mask():
    pass

if __name__ == "__main__":
    from pandas_plink import read_plink1_bin ##Put here so we can load this file as a module on mcluster11 which does not have pandas plink_installed
    from LabQueue.qp import qp
    from LabUtils.addloglevels import sethandlers
    from GeneticsPipeline.helpers_genetic import read_status_table
    sethandlers()
    create_snp_to_gene_mask()

