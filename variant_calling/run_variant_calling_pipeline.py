import os
from os.path import isfile, join, isdir
from LabUtils.addloglevels import sethandlers
import numpy as np
import sys
import pandas as pd
import subprocess
from LabData.DataLoaders.PRSLoader import PRSLoader
from scipy import stats
from LabQueue.qp import qp
import json

def make_json_template():
    json_data = {
        "rna_variant_calling.dedup_bams": None,
        "rna_variant_calling.refFasta": "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/genome_hg38_ERCC92.fa",
        "rna_variant_calling.refFastaIndex": "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/genome_hg38_ERCC92.fa.fai",
        "rna_variant_calling.refDict": "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/genome_hg38_ERCC92.dict",
        "rna_variant_calling.knownVcfs": [
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/resources_broad_hg38_v0_Homo_sapiens_assembly38.known_indels.vcf.gz",
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/resources_broad_hg38_v0_Mills_and_1000G_gold_standard.indels.hg38.vcf.gz"
        ],
        "rna_variant_calling.knownVcfsIndices": [
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/resources_broad_hg38_v0_Homo_sapiens_assembly38.known_indels.vcf.gz.tbi",
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/resources_broad_hg38_v0_Mills_and_1000G_gold_standard.indels.hg38.vcf.gz.tbi"
        ],
        "rna_variant_calling.dbSnpVcf": "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/resources_broad_hg38_v0_Homo_sapiens_assembly38.dbsnp138.vcf",
        "rna_variant_calling.dbSnpVcfIndex": "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/resources_broad_hg38_v0_Homo_sapiens_assembly38.dbsnp138.vcf.idx",
        "rna_variant_calling.minConfidenceForVariantCalling": 20
    }
    return json_data

def run_single_batch(batch, fmap, id, json_dir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/jsons/"):
    json_template = make_json_template()
    json_template["rna_variant_calling.dedup_bams"] = list(fmap.loc[batch].values)
    json_fname = f"{json_dir}batch{id}.json"
    with open(json_fname, "w") as f:
        json.dump(json_template, f, indent=4)
    subprocess.call([ "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/variant_calling_script.csh" + " " + json_fname], shell=True)

def run_all_samples(batch_width, fmap):
    res = {}
    batches = np.array_split(fmap.index.values, len(fmap.index.values)//batch_width)
    with qp(jobname="zach", _mem_def=256, _trds_def=64) as q:
        q.startpermanentrun()
        for id, batch in enumerate(batches):
            res[tuple(batch)] = q.method(run_single_batch, (batch, fmap, id))
        for k, v in res.items():
            try:
                res[k] = q.waitforresult(v)
            except Exception:
                print("Job fell off: ", k)

def crawl_single_rundir(rundir):
    samples = {}
    samples_run = [x for x in os.listdir(rundir) if isdir(rundir + "/" + x)]
    for sample in samples_run:
        this_sample_files = os.listdir(rundir + "/" + sample)
        bam_file = list(filter(lambda x: x.endswith(".bam") and not x.endswith(".bam.bai"), this_sample_files))
        bam_exists = len(bam_file) > 0
        if bam_exists:
            samples[sample] = rundir + "/" + sample + "/" + bam_file[0]
    return pd.Series(samples)

def make_RNA_fmap(basedir):
    fmap = {}
    rundirs = [x for x in os.listdir(basedir) if isdir(basedir + "/" + x) and ("2023" in x or "2024" in x or "2025" in x)]
    for rundir in rundirs:
        is_valid_run =  "5_star_output" in os.listdir(basedir +  "/"  +rundir)
        if is_valid_run:
            fmap[rundir] = crawl_single_rundir(basedir + "/" + rundir + "/5_star_output/")
    fmap = pd.concat(fmap.values())
    fmap = fmap.astype(str)
    fmap.index.name = "SampleName"
    return fmap

def make_merge_fmap(merge_basedir):
    gz_files = {}
    tbi_files = {}
    runs = [x for x in os.listdir(merge_basedir) if isdir(merge_basedir + x)]
    for run in runs:
        run_stages = [x for x in os.listdir(merge_basedir + run) if isdir(merge_basedir + run + "/" + x)]
        if "call-HaplotypeCaller" in run_stages:
            shard_dirs =  [x for x in os.listdir(merge_basedir + run + "/" + "call-HaplotypeCaller/")]
            for shard in shard_dirs:
                gz_file = [x for x in os.listdir(merge_basedir + run + "/" + "call-HaplotypeCaller/" + shard + "/" + "execution") if x.endswith(".gz")]
                tbi_file = [x for x in os.listdir(merge_basedir + run + "/" + "call-HaplotypeCaller/" + shard + "/" + "execution") if x.endswith(".gz.tbi")]
                if len(gz_file) > 0 and len(tbi_file) > 0:
                    sample_id = gz_file[0].split("Aligned")[0]
                    gz_files[sample_id] = merge_basedir + run + "/" + "call-HaplotypeCaller/" + shard + "/" + "execution/" + gz_file[0]
                    tbi_files[sample_id] = merge_basedir + run + "/" + "call-HaplotypeCaller/" + shard + "/" + "execution/" +  tbi_file[0]
    gz_files = pd.Series(gz_files)
    tbi_files = pd.Series(tbi_files)
    merge_fmap = pd.concat([gz_files, tbi_files], axis = 1, ignore_index = False)
    merge_fmap.columns = ["gz_location", "tbi_locaion"]
    return merge_fmap

def make_merge_json_template():
    json_data = {
    "combine_gvcfs.chunked_gvcfs": None,
    "combine_gvcfs.chunked_indices": None,
    "combine_gvcfs.refFasta": "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/genome_hg38_ERCC92.fa",
    "combine_gvcfs.refFastaIndex": "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/genome_hg38_ERCC92.fa.fai",
    "combine_gvcfs.refDict": "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/genome_hg38_ERCC92.dict",
    "combine_gvcfs.output_name": "rna_cohort.g.vcf.gz"}
    return json_data

def merge_all(json_fname):
    subprocess.call(["/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/merge_vcf_script.csh" + " " + json_fname], shell=True)

def merge_all_samples(merge_fmap):
    res = {}
    chunk_size = 240
    gvcf_paths = list(merge_fmap.iloc[:, 0].values)
    index_paths = list(merge_fmap.iloc[:, 1].values)
    # Ensure at least one chunk, avoid ZeroDivisionError
    num_chunks = max(1, len(gvcf_paths) // chunk_size)
    # Use np.array_split() safely and convert to Python lists
    chunked_gvcfs = [chunk.tolist() for chunk in np.array_split(gvcf_paths, num_chunks)]
    chunked_indices = [chunk.tolist() for chunk in np.array_split(index_paths, num_chunks)]
    # Define output paths
    out_gvcfs_json = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/out_gvcfs.json"
    out_idxs_json = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/out_indices.json"
    # Write chunked arrays to JSON, then load from there into the merged one.
    with open(out_gvcfs_json, "w") as out_gvcfs:
        json.dump(chunked_gvcfs, out_gvcfs, indent=2)
    with open(out_idxs_json, "w") as out_idxs:
        json.dump(chunked_indices, out_idxs, indent=2)
    print(f"Wrote {len(chunked_gvcfs)} chunks to '{out_gvcfs_json}' and '{out_idxs_json}'")
    json_template = make_merge_json_template()
    json_fname = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/merge.json"
    with open("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/out_gvcfs.json") as f:
        chunked_gvcfs = json.load(f)
    json_template["combine_gvcfs.chunked_gvcfs"] = chunked_gvcfs
    with open("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/out_indices.json") as f:
        chunked_indices = json.load(f)
    json_template["combine_gvcfs.chunked_indices"] = chunked_indices
    with open(json_fname, "w") as f:
        json.dump(json_template, f, indent=4)
    ##tested on 3k files, and 512 GB RAM was enough
    ##but this requires opening thousands of sample files at once, which will pass the ulimit on smaller queues
    ##we cant set the ulimit above 4096 on the normal q, so use the himem8 q, where the maximum the number of open files is 16384
    ##then we can proceed as normal
    with qp(jobname="zach", _mem_def=10, _trds_def=10, q = ["himem8.q"]) as q:
        q.startpermanentrun()
        res["0"] = q.method(merge_all, (json_fname, ))
        for k, v in res.items():
            try:
                res[k] = q.waitforresult(v)
            except Exception:
                print("Job fell off: ", k)

if __name__ == "__main__":
    sethandlers()
    os.chdir("/net/mraid20/export/mb/logs/")
    run_samples = False
    if run_samples:
        basedir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA"
        fmap = make_RNA_fmap(basedir = basedir)
        run_all_samples(batch_width = 10, fmap = fmap)
    merge = True
    if merge:
        merge_basedir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/cromwell-executions/rna_variant_calling/"
        merge_fmap = make_merge_fmap(merge_basedir = merge_basedir)
        merge_all_samples(merge_fmap)

