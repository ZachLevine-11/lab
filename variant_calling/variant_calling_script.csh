#!/bin/csh

# Activate conda environment
cd /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling
conda activate gatk
 ##force the right version of java for cromwell to work
 ##we do this by source the c shell header file
source ~/.cshrc
##so that gatk can be found inside the image
export PATH=/gatk/gatk:$PATH
java -jar cromwell-87.jar run wdl/rna_variant_calling.wdl --inputs $1
