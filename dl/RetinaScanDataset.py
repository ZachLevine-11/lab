import numpy as np
import pandas as pd
import torchvision
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
 #3000 #tested with 60 GB Ram (1000, batch size 32), (2k, batch size 8), (3k, batch size 2)

def read_fileset():
    fs = pd.read_csv("/net/mraid20/export/jasmine/zach/dl/fundus.csv")
    fs["RegistrationCode"] = list(map(lambda id: "10K_" + str(id), fs.participant_id))
    fs = fs.drop_duplicates(subset = ["RegistrationCode"], keep = "last").set_index("RegistrationCode")
    return fs

class RetinaScanDataset(Dataset):
    def __init__(self, ids, imsize, side = "left"):
        self.side = side
        self.basepath = "/net/mraid20/export/genie/LabData/Data/Pheno/fundus/10k/"
        self.imgsize = imsize
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize([self.imgsize, self.imgsize]),
            torchvision.transforms.ToTensor()])
        self.ids = ids
        self.fileset = read_fileset().loc[self.ids,:]

    def __len__(self):
        return len(self.fileset)

    def get_image(self, idx):
        imgpath = self.fileset.loc[self.fileset.index.get_level_values(0) == idx, "fundus_image_" + self.side].values[
            -1]  ##Just select the last visit for now
        img = io.imread(self.basepath + imgpath)
        return self.transform(img)

    def __getitem__(self, numid):
        idx = self.fileset.index.get_level_values(0)[numid]
        return self.get_image(idx)

def make_train_test_split_retina(train_ids_retina, test_ids_retina, imsize, side = "left"):
    ds_train = RetinaScanDataset(train_ids_retina, imsize, side)
    RetinaScanLoader_train = DataLoader(ds_train, batch_size = 2, shuffle=True)
    ds_test = RetinaScanDataset(test_ids_retina, imsize, side)
    RetinaScanLoader_test = DataLoader(ds_test, batch_size=1, shuffle=True)
    return ds_train, RetinaScanLoader_train, ds_test, RetinaScanLoader_test
