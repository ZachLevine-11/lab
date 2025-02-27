import torch
from torch.utils.data import Dataset
from pandas_plink import read_plink1_bin
from GeneticsPipeline.helpers_genetic import read_status_table
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader

class GeneticsDataset(Dataset):
    def __init__(self):
        self.status_table = read_status_table()
        try:
            self.status_table = self.status_table[self.status_table.passed_qc].copy()
        except ValueError:  ##In case the genetics pipeline is running
            self.status_table = self.status_table.dropna()
            self.status_table = self.status_table[self.status_table.passed_qc].copy()
        ##Use the ld pruned genetics files
        self.binaries = read_plink1_bin(
            bed="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_extra_qc_extra_before_king.bed",
            bim="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_extra_qc_extra_before_king.bim",
            fam="/net/mraid20/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_extra_qc_extra_before_king.fam")
        self.binaries["sample"] = self.binaries.sample.to_pandas().apply(
            self.status_table.set_index('gencove_id').RegistrationCode.to_dict().get)
        self.bodymeasures = BodyMeasuresLoader().get_data(study_ids="10K").df

    def __len__(self):
        return len(self.binaries)

    def __getitem__(self, tenK_id):
        return torch.from_numpy(self.binaries.sel(sample=tenK_id).to_pandas().values), \
               torch.Tensor(self.bodymeasures.loc[self.bodymeasures.index.get_level_values(0) == tenK_id, "height"])[
                   0]  ##index at zero just incase there are duplicate entries
