import os
from os.path import isfile, join, isdir
from LabUtils.addloglevels import sethandlers
import numpy as np
import pandas as pd
import subprocess
from LabData.DataLoaders.PRSLoader import PRSLoader
import json
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

def make_merge_fmap():
    vcfs = {}
    tbis = {}
    rundirs = [x for x in os.listdir("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/cromwell-executions/rna_variant_calling/")]
    for rundir in rundirs:
        shards = os.listdir("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/cromwell-executions/rna_variant_calling/" + rundir + "/call-HaplotypeCaller/")
        for shard in shards:
            shardfiles = os.listdir("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/cromwell-executions/rna_variant_calling/" + rundir + "/call-HaplotypeCaller/" + shard + "/execution/")
            tbi_file = list(filter(lambda x: x.endswith(".tbi"), shardfiles))
            vcf_file = list(filter(lambda x: x.endswith(".vcf.gz"), shardfiles))
            if len(vcf_file) > 0 and len(tbi_file) > 0:
                path_prefix = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/cromwell-executions/rna_variant_calling/" + rundir + "/call-HaplotypeCaller/" + shard + "/execution/"
                vcf_file  = path_prefix + vcf_file[0]
                tbi_file = path_prefix + tbi_file[0]
                person_vcf = vcf_file.split("Aligned")[0]
                person_tbi = tbi_file.split("Aligned")[0]
                if person_vcf == person_tbi:
                    vcfs[person_vcf] = vcf_file
                    tbis[person_vcf] = tbi_file
    vcfs = pd.Series(vcfs)
    tbis = pd.Series(tbis)
    fmap = pd.concat([vcfs, tbis], axis = 1)
    fmap.index.name = "SampleName"
    fmap = fmap.astype(str)
    fmap.index.name = "SampleName"
    fmap.columns = ["vcf_file_loc", "tbi_file_loc"]
    return fmap

def make_merge_template():
    json_data = {
    "combine_gvcfs.chunked_gvcfs": None,
     "combine_gvcfs.chunked_indices": None,
    "combine_gvcfs.refFasta": "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/genome_hg38_ERCC92.fa",
    "combine_gvcfs.refFastaIndex": "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/genome_hg38_ERCC92.fa.fai",
    "combine_gvcfs.refDict": "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/data/genome_hg38_ERCC92.dict",
    "combine_gvcfs.output_name": "rna_cohort.g.vcf.gz"
    }
    return json_data


def merge_all_samples(merge_fmap, batch_size = 32):
    res = {}
    template = make_merge_template()
    vcfs = merge_fmap.iloc[:, 0]
    vcfs_list = vcfs.values.tolist()
    # Split the list into batches
    batched_data = [vcfs_list[i:i + batch_size] for i in range(0, len(vcfs_list), batch_size)]
    template["combine_gvcfs.chunked_gvcfs"] = batched_data
    tbis = merge_fmap.iloc[:, 1]
    tbis_list = tbis.values.tolist()
    # Split the list into batches
    batched_data_tbis = [tbis_list[i:i + batch_size] for i in range(0, len(tbis_list), batch_size)]
    template["combine_gvcfs.chunked_indices"] = batched_data_tbis
    with open("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/merge.json", "w") as f:
        json.dump(template, f, indent=4)
    print(f"Wrote {len(batched_data_tbis)} batches")
    with qp(jobname="zach", _mem_def=500, _trds_def=110, q = ["himem8.q"]) as q:
        q.startpermanentrun()
        res["0"] = q.method(lambda: subprocess.call(["/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/merge_vcf_script.csh"], shell=True), ())
        for k, v in res.items():
            try:
                res[k] = q.waitforresult(v)
            except Exception:
                print("Job fell off: ", k)

if __name__ == "__main__":
    sethandlers()
    do_variants = False
    if do_variants:
        os.chdir("/net/mraid20/export/mb/logs/")
        basedir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA"
        fmap = make_RNA_fmap(basedir = basedir)
        run_all_samples(batch_width = 10, fmap = fmap)
    do_merge = True
    if do_merge:
        os.chdir("/net/mraid20/export/mb/logs/")
        merge_fmap = make_merge_fmap()
        merge_all_samples(batch_size = 32, merge_fmap = merge_fmap)


