import numpy as np
import pandas as pd
import logging
import os
from os.path import isfile, join
from GeneticsPipeline.helpers_genetic import read_status_table, run_plink2, required_memory_gb
from LabUtils.Scripts.shell_commands_execute import ShellCommandsExecute
from LabUtils.addloglevels import sethandlers
from GeneticsPipeline.config import plink19_bin, plink2_bin, plink_prs_target_prefix
from scores_work import get_all_result_files

sethandlers()


def make_plink2_command(score_file_name, SOMAscan=True):
    if SOMAscan: whichDir = "SOMAscan/"
    cmd = plink2_bin + ' --bfile /net/mraid20/export/genie/10K/genetics/Gencove/allsamples_qc --score /net/mraid20/export/jasmine/zach/scores/score_files/' + whichDir + score_file_name + ' 1 4 6 header list-variants cols=scoresums --out /net/mraid20/export/jasmine/zach/scores/score_results/' + whichDir + score_file_name + ' --covar /net/mraid20/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt --covar-col-nums 3-14 --covar-variance-standardize'
    return cmd


##use redundant to exclude ones we've already done
def get_all_score_files(SOMAscan=True, cooldir="/net/mraid20/export/jasmine/zach/scores/score_files/", redundant=False):
    if SOMAscan: allDir = cooldir + "SOMAscan/"
    onlyfiles = [f for f in os.listdir(allDir) if isfile(join(allDir, f))]
    if not redundant: onlyfiles = list(set(onlyfiles) - set(
        map(lambda x: x.split(".sscore")[0], get_all_result_files())))  ##Don't redo scores we've already done
    return onlyfiles


def do_all_scores(SOMAscan=True):
    os.chdir("/net/mraid20/export/mb/logs/")
    all_files = get_all_score_files(SOMAscan=SOMAscan)
    for score_file_name in all_files:
        run_plink2(make_plink2_command(score_file_name, SOMAscan=SOMAscan), score_file_name,
                   required_memory_gb("/net/mraid20/export/genie/10K/genetics/Gencove/allsamples_qc.bed"))
    print("Ran all pqtl generation commands")
