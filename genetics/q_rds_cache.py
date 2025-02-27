from LabQueue.qp import qp
import pandas as pd
import subprocess
import os
from os.path import isfile, join
from LabUtils.addloglevels import sethandlers
from scores_work import stack_matrices_and_bonferonni_correct
from LabData.DataLoaders.PRSLoader import PRSLoader

def single_cache_job(pheno):
    subprocess.call(["~/PycharmProjects/genetics/q_single_cache_job.sh" + " " + pheno], shell=True)

#only works in debug mode
def generate_rds_cache(results_dir = "/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_results/"):
    os.chdir("/net/mraid20/export/mb/logs/")
    with qp(jobname = "rds", delay_batch = 30) as q: ##genie58 doesn't have Rstudio installed, so use this to avoid sending jobs there
        q.startpermanentrun()
        files = [f for f in os.listdir(results_dir) if isfile(join(results_dir, f)) and f.endswith(".glm.linear") and f.startswith("batch0.")]
        fundict = {}
        for file in files:
            operative_file = file.split("batch0.")[-1].split(".glm.linear")[0].replace("|", "XPIPEX") ##to avoid pipes breaking the queue shell command
            fundict[operative_file] = q.method(single_cache_job, (operative_file,))
        for k, v in fundict.items():
                try:
                    fundict[k] = q.waitforresult(v)
                except Exception:
                    fundict[k] = None

def make_prs_assoc_table():
    s = stack_matrices_and_bonferonni_correct(fillwithNA=False)
    thedict = PRSLoader().get_data().df_columns_metadata
    thedict.index = list(map(lambda thestr: "pvalue_" + thestr, thedict.index))
    thedict = thedict.h2_description.to_dict()
    stackmat = s.rename(dict(zip(list(s.columns),
                                 [thedict.get(col) for col in s.columns])), axis=1)
    mapper = list(map(lambda potential: type(potential) == str, stackmat.columns))
    s = stackmat.loc[:, mapper]
    s.to_csv("/home/zacharyl/gwasInterface/inst/prs_assoc_corrected_clinical.csv")

if __name__ == "__main__":
    sethandlers()
    generate_rds_cache(results_dir = "/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/microbiome/gwas_results_mb/")