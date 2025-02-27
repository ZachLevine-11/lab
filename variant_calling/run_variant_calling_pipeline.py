import os
from os.path import isfile, join, isdir
from LabUtils.addloglevels import sethandlers
import numpy as np
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
    with qp(jobname="zach", _mem_def=256, _trds_def=32) as q:
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


if __name__ == "__main__":
    sethandlers()
    os.chdir("/net/mraid20/export/mb/logs/")
    basedir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA"
    fmap = make_RNA_fmap(basedir = basedir)
    run_all_samples(batch_width = 10, fmap = fmap)
