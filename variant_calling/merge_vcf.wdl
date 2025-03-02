version 1.0

################################################################################
# TASK: CombineGVCFs
# Combines an array of GVCF files into a single multi-sample GVCF (single-threaded).
################################################################################
task CombineGVCFs {
    input {
        Array[File] gvcf_files
        Array[File] gvcf_indices
        File ref_fasta
        File ref_fasta_index
        File ref_dict
        String output_name
        String memory_gb = "20"
    }

    command <<<
        set -e

        # Optional: ensure read perms on the input files
        chmod ugo+r ~{sep=" " gvcf_files}
        chmod ugo+r ~{sep=" " gvcf_indices}

        # Put the spaced list of GVCFs into a shell variable
        # e.g. "file1.vcf.gz file2.vcf.gz file3.vcf.gz"
        GVCF_LIST="~{sep=" " gvcf_files}"

        # Run GATK CombineGVCFs:
        # Expand GVCF_LIST in a small for-loop to produce:
        #   --variant file1.vcf.gz --variant file2.vcf.gz ...
        /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/gatk-4.6.1.0/gatk \
            --java-options "-Xmx~{memory_gb}g" \
            CombineGVCFs \
            -R ~{ref_fasta} \
            $(for f in $GVCF_LIST; do echo -n "--variant $f "; done) \
            -O ~{output_name}
    >>>

    output {
        File output_vcf       = output_name
        File output_vcf_index = output_name + ".tbi"
    }

    runtime {
        cpu: 1
        memory: "~{memory_gb} GB"
        singularity: "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/gatk_latest.sif"
    }
}


################################################################################
# TASK: IndexGVCF
# Creates a .tbi index for a given VCF using GATK IndexFeatureFile.
################################################################################
task IndexGVCF {
    input {
        File vcf
    }
    command <<<
        /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/gatk-4.6.1.0/gatk \
            IndexFeatureFile \
            -I ~{vcf}
    >>>
    output {
        File vcf_out = vcf
        File index = vcf + ".tbi"
    }
    runtime {
        cpu: 1
        memory: "4 GB"
        singularity: "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/gatk_latest.sif"
    }
}


################################################################################
# WORKFLOW: combine_gvcfs
# Expects pre-chunked arrays of GVCFs and indices in JSON form.
################################################################################
workflow combine_gvcfs {
    input {
        # Instead of a single Array[File], you provide an Array of Arrays: chunked GVCFs
        Array[Array[File]] chunked_gvcfs
        Array[Array[File]] chunked_indices

        File  refFasta
        File  refFastaIndex
        File  refDict

        String output_name = "cohort.g.vcf.gz"
        String combine_memory_gb = "20"
    }

    # Scatter over each chunk in parallel
    scatter (i in range(length(chunked_gvcfs))) {
        Array[File] chunk_gvcfs   = chunked_gvcfs[i]
        Array[File] chunk_indices = chunked_indices[i]

        call CombineGVCFs as combineBatch {
            input:
                gvcf_files       = chunk_gvcfs,
                gvcf_indices     = chunk_indices,
                ref_fasta        = refFasta,
                ref_fasta_index  = refFastaIndex,
                ref_dict         = refDict,
                output_name      = "batch_" + i + ".g.vcf.gz",
                memory_gb        = combine_memory_gb
        }
    }

    # If multiple chunks exist, combine them all into one final GVCF
    if (length(chunked_gvcfs) > 1) {
        call CombineGVCFs as combineAll {
            input:
                gvcf_files       = combineBatch.output_vcf,
                gvcf_indices     = combineBatch.output_vcf_index,
                ref_fasta        = refFasta,
                ref_fasta_index  = refFastaIndex,
                ref_dict         = refDict,
                output_name      = output_name,
                memory_gb        = combine_memory_gb
        }
    }

    # Final step: index the final merged VCF
    call IndexGVCF as indexFinal {
        input:
            vcf = select_first([
                combineAll.output_vcf,
                combineBatch.output_vcf[0]
            ])
    }

    # Final workflow outputs
    output {
        File final_gvcf = indexFinal.vcf_out
        File final_gvcf_index = indexFinal.index
    }
}
