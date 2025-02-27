import os
from os import path
import shutil

def crawl_single_run_dir(rundir, step, final_dir):
    newdir = rundir + step + "/"
    run_dirs = [newdir + x for x in os.listdir(newdir) if not path.isfile(x)]
    for sample in run_dirs:
        if len(os.listdir(sample)) == 0:
            continue
        else:
            bamfile = os.listdir(sample)[0]
            entire_filepath =  sample + "/" + bamfile
            shutil.copyfile(entire_filepath, final_dir + bamfile)

if __name__ == "__main__":
    basedir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/"
    final_dir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/gatk/inputs/HPP_RNASeq/"
    step = "6_move_bc"
    rundirs = [basedir + x + "/" for x in os.listdir(basedir) if not path.isfile(x) and str(x).endswith("2") or str(x).endswith("3") or str(x).endswith("4")]
    crawl_single_run_dir(rundirs[0], step, final_dir)

##it is important to source the c shell header file to use the right version of java for gatk
shell_cmds = """
conda activate gatk
source ~/.cshrc
cd /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/gatk/gatk-4.6.0.0
java -jar cromwell-33.1.jar run /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/gatk/gatk4-rnaseq-germline-snps-indels/gatk4-rna-best-practices.wdl --inputs /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/gatk/gatk4-rnaseq-germline-snps-indels/gatk4-rna-germline-variant-calling.inputs.json
"""