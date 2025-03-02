#! /bin/csh

# Activate conda environment
cd /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling
conda activate gatk
 ##force the right version of java for cromwell to work
 ##we do this by source the c shell header file
source ~/.cshrc
##so that gatk can be found inside the image
export PATH=/gatk/gatk:$PATH
##need to force a specific conf to use wdl 1.0 explicitly here
java -Dconfig.file=./cromwell.conf -jar cromwell-87.jar run wdl/merge_vcf.wdl --inputs merge.json