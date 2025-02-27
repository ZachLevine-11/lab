#!/bin/csh

cd /net/mraid20/export/jasmine/zach/height_gwas/all_gwas/ldsc/ldsc-master
conda activate ldsc
ldsc-master/ldsc.py --rg $1 --ref-ld-chr $2 --w-ld-chr $3 --out $4 --no-check-alleles
conda deactivate

