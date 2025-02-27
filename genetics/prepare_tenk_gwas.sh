#!/bin/csh

cd /net/mraid20/export/jasmine/zach/height_gwas/all_gwas/ldsc/ldsc
conda activate ldsc
python munge_sumstats.py --sumstats $1 --out $2 --a1-inc --a2 AX --merge-alleles /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/height_gwas/all_gwas/ldsc/eur_w_ld_chr/w_hm3.snplist --snp ID --N-col OBS_CT --chunksize 500000
conda deactivate