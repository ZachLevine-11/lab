version 1.0

workflow rna_variant_calling {
    input {
        Array[File] dedup_bams
        File refFasta
        File refFastaIndex
        File refDict
        Array[File] knownVcfs
        Array[File] knownVcfsIndices
        File dbSnpVcf
        File dbSnpVcfIndex
        Int minConfidenceForVariantCalling
    }

    scatter (dedup_bam in dedup_bams) {
        call AddOrReplaceReadGroups {
            input:
                input_bam = dedup_bam,
                sample_name = basename(dedup_bam, ".BAM")
        }

        call SplitNCigarReads {
            input:
                input_bam       = AddOrReplaceReadGroups.output_bam,
                base_name       = basename(dedup_bam, ".BAM") + ".split",
                ref_fasta       = refFasta,
                ref_fasta_index = refFastaIndex,
                ref_dict        = refDict
        }

        call BaseRecalibrator {
            input:
                input_bam                   = SplitNCigarReads.output_bam,
                input_bam_index             = SplitNCigarReads.output_bam_index,
                recal_output_file           = basename(dedup_bam, ".BAM") + ".recal_data.csv",
                dbSNP_vcf                   = dbSnpVcf,
                dbSNP_vcf_index             = dbSnpVcfIndex,
                known_indels_sites_VCFs     = knownVcfs,
                known_indels_sites_indices  = knownVcfsIndices,
                ref_fasta                   = refFasta,
                ref_fasta_index             = refFastaIndex,
                ref_dict                    = refDict
        }

        call ApplyBQSR {
            input:
                input_bam            = SplitNCigarReads.output_bam,
                input_bam_index      = SplitNCigarReads.output_bam_index,
                recalibration_report = BaseRecalibrator.recalibration_report,
                base_name            = basename(dedup_bam, ".BAM") + ".aligned.duplicates_marked.recalibrated",
                ref_fasta            = refFasta,
                ref_fasta_index      = refFastaIndex,
                ref_dict             = refDict
        }

        call HaplotypeCaller {
            input:
                input_bam       = ApplyBQSR.output_bam,
                input_bam_index = ApplyBQSR.output_bam_index,
                base_name       = basename(dedup_bam, ".BAM") + ".hc",
                ref_fasta       = refFasta,
                ref_fasta_index = refFastaIndex,
                ref_dict        = refDict,
                dbSNP_vcf       = dbSnpVcf,
                dbSNP_vcf_index = dbSnpVcfIndex,
                stand_call_conf = minConfidenceForVariantCalling
        }
    }
}

task SplitNCigarReads {
    input {
        File input_bam
        String base_name
        File ref_fasta
        File ref_fasta_index
        File ref_dict
    }

    command <<<
        set -e
        /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/gatk-4.6.1.0/gatk SplitNCigarReads \
            -R ~{ref_fasta} \
            -I ~{input_bam} \
            -O ~{base_name}.bam
    >>>

    output {
        File output_bam       = "~{base_name}.bam"
        File output_bam_index = "~{base_name}.bai"
    }

    runtime {
        singularity: "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/gatk_latest.sif"
        memory: "32 GB"
        cpu: 8
    }
}

task AddOrReplaceReadGroups {
    input {
        File input_bam
        String sample_name
    }

    command <<<
        set -e
        /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/gatk-4.6.1.0/gatk AddOrReplaceReadGroups \
            -I ~{input_bam} \
            -O ~{sample_name}.rg.bam \
            -RGID flowcell1 \
            -RGLB lib1 \
            -RGPL ILLUMINA \
            -RGPU unit1 \
            -RGSM ~{sample_name} \
            --CREATE_INDEX true
    >>>

    output {
        File output_bam = "~{sample_name}.rg.bam"
        File output_bai = "~{sample_name}.rg.bai"
    }

    runtime {
        singularity: "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/gatk_latest.sif"
        memory: "32 GB"
        cpu: 8
    }
}

task BaseRecalibrator {
    input {
        File input_bam
        File input_bam_index
        String recal_output_file
        File dbSNP_vcf
        File dbSNP_vcf_index
        Array[File] known_indels_sites_VCFs
        Array[File] known_indels_sites_indices
        File ref_dict
        File ref_fasta
        File ref_fasta_index
    }

    command <<<
        set -e
        /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/gatk-4.6.1.0/gatk BaseRecalibrator \
            -I ~{input_bam} \
            -R ~{ref_fasta} \
            --known-sites ~{dbSNP_vcf} \
            --known-sites ~{sep=" --known-sites " known_indels_sites_VCFs} \
            -O ~{recal_output_file}
    >>>

    output {
        File recalibration_report = "~{recal_output_file}"
    }

    runtime {
        singularity: "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/gatk_latest.sif"
        memory: "32 GB"
        cpu: 8
    }
}

task ApplyBQSR {
    input {
        File input_bam
        File input_bam_index
        String base_name
        File recalibration_report
        File ref_dict
        File ref_fasta
        File ref_fasta_index
    }

    command <<<
        set -e
       /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/gatk-4.6.1.0/gatk ApplyBQSR \
            -R ~{ref_fasta} \
            -I ~{input_bam} \
            --bqsr-recal-file ~{recalibration_report} \
            -O ~{base_name}.bam
    >>>

    output {
        File output_bam       = "~{base_name}.bam"
        File output_bam_index = "~{base_name}.bai"
    }

    runtime {
        singularity: "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/gatk_latest.sif"
        memory: "32 GB"
        cpu: 8
    }
}

task HaplotypeCaller {
    input {
        File input_bam
        File input_bam_index
        String base_name
        File ref_dict
        File ref_fasta
        File ref_fasta_index
        File dbSNP_vcf
        File dbSNP_vcf_index
        Int stand_call_conf
    }

    command <<<
        set -e
        /net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/rna_variant_calling/gatk-4.6.1.0/gatk  HaplotypeCaller \
            -R ~{ref_fasta} \
            -I ~{input_bam} \
            -O ~{base_name}.vcf.gz \
            -dont-use-soft-clipped-bases \
            --standard-min-confidence-threshold-for-calling ~{stand_call_conf} \
            --dbsnp ~{dbSNP_vcf} \
            --emit-ref-confidence GVCF \
            --native-pair-hmm-threads 8 \
            --max-alternate-alleles 3 \
            --disable-read-filter NotDuplicateReadFilter
    >>>

    output {
        File output_vcf       = "~{base_name}.vcf.gz"
        File output_vcf_index = "~{base_name}.vcf.gz.tbi"
    }

    runtime {
        singularity: "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/variants/gatk_latest.sif"
        memory: "32 GB"
        cpu: 8
    }
}
