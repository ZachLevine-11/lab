#!/bin/bash

##run on cluster11
##requries 60GB of RAM


##cd to the directory with the image file
cd /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/better4u

##start the singularity with explicit bindings to any directories it will need
singularity run --bind /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/:/mnt/data --net --network none ./better4u_latest.sif

##plink het calc
plink \
--memory 51200 \
--bfile /mnt/data/height_gwas/all_gwas/Genetics_Backups/allsamples \
--out /mnt/data/better4u/allsamples --het

##Rscript they asked for
Rscript \
-e '{
hetData <- read.table("allsamples.het",header=TRUE,check.names=FALSE)
rownames(hetData) <- hetData$IID
heterozygosity <- 1 - hetData[,3]/hetData[,5]
names(heterozygosity) <- rownames(hetData)
avg <- median(heterozygosity,na.rm=TRUE)
dev <- IQR(heterozygosity,na.rm=TRUE)
keep <- heterozygosity > avg - 3*dev & heterozygosity < avg + 3*dev
goodHet <- rownames(hetData)[keep]
6
write.table(hetData[keep,c("FID","IID"),drop=FALSE],
file="het_samples_pass.txt",col.names=FALSE,row.names=FALSE,quote=FALSE)
}'

##First set of QC filters
plink \
--memory 51200 \
--bfile /mnt/data/height_gwas/all_gwas/Genetics_Backups/allsamples \
--out  /mnt/data/better4u/allsamples_tmp  \
--make-bed \
--geno 0.02 \
--maf 0.05 \
--hwe 0.000001 \
--mind 0.05

##another command
cut -d" " -f1-2 allsamples_tmp.fam > generic_samples_pass.txt
cut -f2 allsamples_tmp.bim > generic_variants_pass.txt

##and another one
Rscript \
-e '{
het <- read.table("het_samples_pass.txt")
rownames(het) <- het[,2]
gen <- read.table("generic_samples_pass.txt")
rownames(gen) <- gen[,2]
pass <- intersect(rownames(het),rownames(gen))
write.table(gen[pass,,drop=FALSE],file="all_samples_pass.txt",
col.names=FALSE,row.names=FALSE,quote=FALSE)
}'

##another one
plink \
--memory 55200 \
--bfile /mnt/data/height_gwas/all_gwas/Genetics_Backups/allsamples \
--out /mnt/data/better4u/allsamples_filtered \
--extract generic_variants_pass.txt \
--keep all_samples_pass.txt \
--make-bed

plink \
--bfile /mnt/data/better4u/allsamples_filtered \
--out /mnt/data/better4u/allsamples_filtered \
--indep-pairwise 50 5 0.2

plink \
--bfile /mnt/data/better4u/allsamples_filtered \
--out /mnt/data/better4u/allsamples_filtered \
--genome gz \
--exclude /mnt/data/better4u/allsamples_filtered.prune.out


Rscript \
-e '{
fam <- read.table("allsamples_filtered.fam")
ibdCoeff <- read.table("allsamples_filtered.genome.gz",header=TRUE)
ibdCoeff <- ibdCoeff[ibdCoeff$PI_HAT>=0.5,,drop=FALSE]
if (nrow(ibdCoeff) > 0) {
bad <- unique(c(ibdCoeff$IID1,ibdCoeff$IID2))
ii <- match(bad,fam[,2])
write.table(fam[ii,c(1,2),drop=FALSE],file="ibd_samples_remove.txt",
col.names=FALSE,row.names=FALSE,quote=FALSE)
}
}'

plink \
--bfile allsamples_filtered \
--out allsamples_ibd \
--remove ibd_samples_remove.txt \
--make-bed

king \
-b allsamples_filtered.bed \
--unrelated \
--degree 2 \
--prefix individuals_

##the snp extract list in rsid format
cd /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/better4u/rsid_info
> all_100G_snps_with_rsid.txt
for file in *.txt; do
    # Extract the second column using awk and append it to the output file
    awk '{print $2}' "$file" >> all_100G_snps_with_rsid.txt
done
echo "Stacked column created in all_100G_snps_with_rsid.txt"
cd /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/better4u/
mv rsid_info/all_100G_snps_with_rsid.txt ./

##also stack the mapping
# Initialize the output file
cd /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/better4u/rsid_info
> all_100G_snps_with_both.txt
# Loop through all .txt files in the current directory
for file in *.txt; do
    # Append both columns from the current file to the output file
    cat "$file" >> all_100G_snps_with_both.txt
done
echo "Stacked columns created in all_100G_snps_with_both.txt"
cd /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/better4u/
mv rsid_info/all_100G_snps_with_both.txt ./

##map the refpos a1 file to use rsids
# Create a mapping of chr:pos:ref:alt to rsid from all_100g_snps_with_both.txt
awk '{map[$1]=$2} END {for (key in map) print key, map[key]}' all_100g_snps_with_both.txt > snp_map.txt
# Replace chr:pos:ref:alt in refpos_1000g.txt with corresponding rsid
awk 'NR==FNR {snp_map[$1]=$2; next} {if ($1 in snp_map) print snp_map[$1], $2; else print $1, $2}' snp_map.txt refpos_1000g.txt > refpos_1000g_with_rsid.txt

##also map the means and loadings file, but use python, using the snp map we made above

##then force the A1/A2 to match 100G and extract the snps for PCA
plink \
--bfile allsamples_filtered \
--a1-allele refpos_1000g_with_rsid.txt 2 1 \
--out allsamples_mapped_ids_for_PCA \
--extract all_100G_snps_with_rsid.txt \
--make-bed

##do the PCA
flashpca \
--bfile allsamples_mapped_ids_for_PCA \
--inmeansd means_100g_with_rsid.txt \
--inload loads_100g_with_rsid.txt\
--project \
--outproj projections.txt \
--verbose

##LD Pruning after removing based on IBD
plink \
--memory 55200 \
--bfile allsamples_ibd \
--indep 50 5 2 \
--out allsamples_mapped_ids_for_PCA_pruned

##Step 3: Create PLINK files with the pruned variants and all samples
##all samples_filtered was the pruned variants before removed the samples based on ibd
plink \
--memory 55200 \
--bfile allsamples_filtered \
--extract allsamples_mapped_ids_for_PCA_pruned.prune.in \
--out allsamples_mapped_ids_for_PCA_pruned_for_grm \
--make-bed

gcta64 \
--bfile allsamples_mapped_ids_for_PCA_pruned_for_grm \
--make-grm \
--autosome \
--thread-num 48 \
--out allsamples_grm

gcta64 \
--grm allsamples_grm \
--make-bK-sparse 0.05 \
--out allsamples_grm_sparse

##now do each phenotype
gcta64 \
--bfile allsamples_mapped_ids_for_PCA_pruned_for_grm \
--grm-sparse allsamples_grm_sparse \
--est-vg HE \
--pheno features.csv \
--qcovar covars_quantative.csv \
--covar covars_not_quantitative.csv \
--fastGWA-mlm \
--model-only \
--thread-num 4 \
--mpheno 1 \
--out fit_weight

gcta64 \
--bfile allsamples_mapped_ids_for_PCA_pruned_for_grm \
--load-model fit_weight.fastGWA \
--out fit_weight_out \
--thread-num 4

gcta64 \
--bfile allsamples_mapped_ids_for_PCA_pruned_for_grm \
--grm-sparse allsamples_grm_sparse \
--est-vg HE \
--pheno features.csv \
--qcovar covars_quantative.csv \
--covar covars_not_quantitative.csv \
--fastGWA-mlm \
--model-only \
--thread-num 4 \
--mpheno 2 \
--out fit_bmi

gcta64 \
--bfile allsamples_mapped_ids_for_PCA_pruned_for_grm \
--load-model fit_bmi.fastGWA \
--out fit_bmi_out \
--thread-num 4

gcta64 \
--bfile allsamples_mapped_ids_for_PCA_pruned_for_grm \
--grm-sparse allsamples_grm_sparse \
--est-vg HE \
--pheno features.csv \
--qcovar covars_quantative.csv \
--covar covars_not_quantitative.csv \
--fastGWA-mlm \
--model-only \
--thread-num 4 \
--mpheno 3 \
--out 02_baseline_diff_weight & gcta64 \
--bfile allsamples_mapped_ids_for_PCA_pruned_for_grm \
--load-model 02_baseline_diff_weight.fastGWA \
--out 02_baseline_diff_weight \
--thread-num 4

gcta64 \
--bfile allsamples_mapped_ids_for_PCA_pruned_for_grm \
--load-model 02_baseline_diff_weight.fastGWA \
--out 02_baseline_diff_weight_out \
--thread-num 4

gcta64 \
--bfile allsamples_mapped_ids_for_PCA_pruned_for_grm \
--grm-sparse allsamples_grm_sparse \
--est-vg HE \
--pheno features.csv \
--qcovar covars_quantative.csv \
--covar covars_not_quantitative.csv \
--fastGWA-mlm \
--model-only \
--thread-num 4 \
--mpheno 4 \
--out 02_baseline_diff_bmi

 gcta64 \
--bfile allsamples_mapped_ids_for_PCA_pruned_for_grm \
--load-model 02_baseline_diff_bmi.fastGWA \
--out 02_baseline_diff_bmi_out \
--thread-num 4

'''

