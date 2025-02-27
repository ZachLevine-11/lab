##Save a single GWAS file as an .RDS file
phenoName <- commandArgs((trailingOnly = TRUE))[1]
threshold = 5*10**(-5) #-5 for serum metabolites/gut mb, and -3.5 for everything else
thefile <- read.csv(paste0("/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/microbiome/gwas_results_mb/", "batch0.", gsub("XPIPEX", "|", phenoName), ".glm.linear"), sep = "\t")
thefile_sig <- thefile[thefile$P < threshold,]
saveRDS(thefile_sig, paste0("~/gwasInterface/inst/full_results_rds/", gsub("XPIPEX", "|", phenoName), ".glm.linear", ".Rds"))
