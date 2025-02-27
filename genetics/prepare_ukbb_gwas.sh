#!/bin/csh

cd /net/mraid20/export/jasmine/zach/height_gwas/all_gwas/ldsc/ldsc
conda activate ldsc
python munge_sumstats.py --sumstats $1 --out $2 --snp variant --N 361194 --a1 ref_allele --a2 alt_allele -merge-alleles /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/height_gwas/all_gwas/ldsc/w_hm3.snplist  --chunksize 500000
conda deactivate
