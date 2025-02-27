import numpy as np
import pandas as pd
import torch
import os
from torch import nn as nn
from torch import optim
from RetinaScanDataset import make_train_test_split_retina, read_fileset
from ECGDataset import make_train_test_split_ecg
from tcn import compare, peakLoss
from tcn import both as ecg
from sdca import both as eye
import matplotlib.pyplot as plt
from tcn import train_loop as train_ecg
from sdca import train_loop as train_eye

def contrast(eye_embeddings,ecg_embeddings):
    contrast = 0
    temp = 0.9
    for i in range(len(eye_embeddings)): ##torch's nn functional cosine similarity is just a numerically stable version of the dot poduct
        denom = 0
        num = torch.exp(torch.nn.functional.cosine_similarity(eye_embeddings[i], ecg_embeddings[i], dim = -1)/temp)
        for j in range(len(eye_embeddings)):
            if i != j:
                denom += torch.exp(torch.nn.functional.cosine_similarity(ecg_embeddings[j], eye_embeddings[i], dim = -1)/temp)
        fancyterm = torch.log(num / denom)
        contrast += fancyterm
        denom = 0
        for j in range(len(eye_embeddings)):
            if i != j:
             denom += torch.exp(torch.nn.functional.cosine_similarity(ecg_embeddings[i], eye_embeddings[j], dim = -1)/temp)
        fancyterm = torch.log(num / denom)# the eps is for numerical stability so we don't take the power of a negative number
        contrast += fancyterm
    return -contrast

def combined_loss(x,y,eye_reconst_ecg, eye_original, ecg_reconst, ecg_original, alpha_ecg = 0, alpha_eye = 0, star = 1, device = None):
    loss = torch.complex(torch.Tensor([0]), torch.Tensor([0])).to(device)
    ecg_error = torch.complex(torch.Tensor([0]), torch.Tensor([0])).to(device)
    temp = torch.complex(torch.Tensor([0]), torch.Tensor([0])).to(device)
    if alpha_eye > 0:
        for i in range(len(eye_reconst_ecg)): ##torch's nn functional cosine similarity is just a numerically stable version of the dot poduct
            eye_error = nn.MSELoss()(eye_reconst_ecg[i], eye_original[i])
            loss += alpha_eye*eye_error ##balcance ecg and eye reconst terms
    justeye = loss.item()
    if star > 0:
        temp = star*contrast(x,y)
        loss += temp
    if alpha_ecg > 0:
        #the loop is built in to the loss function
        ecg_error = peakLoss(ecg_reconst, ecg_original, alpha=0, beta=0, gamma=0,  delta=1, theta=1/400000)
        loss += alpha_ecg*ecg_error
    print("eye: ", justeye) ##already included
    print("ecg: ", alpha_ecg*ecg_error.item())
    print("contrast: ", temp.item())
    return loss

def batch_manager(ids_regCode, ds_train_ecg, ds_train_retina, mapper, device, imsize):
    ids_ecg = list(map(lambda x: mapper.get(x), ids_regCode))
    batched_ecgs = torch.zeros([len(ids_regCode), 12, 10000], device = device)
    batched_retinas = torch.zeros([len(ids_regCode), 3, imsize, imsize], device = device)
    for i in range(min(len(ids_regCode), len(ids_regCode))):
        ##getitem only understands the integer index in the dataset
        int_id_ecg = list(ds_train_ecg.uniqueids).index(ids_ecg[i])
        ecg = ds_train_ecg.__getitem__(int_id_ecg)
        ecg = (ecg - ecg.mean(dim=1, keepdim=True)) / ecg.std(dim=1, keepdim=True)
        ecg = ecg.permute(1, 0)  ##this is a single person's time series, so there is no batch dimension
        batched_ecgs[i, :, :, ] = ecg
        int_id_retina = ds_train_retina.fileset.index.get_loc(ids_regCode[i])
        retina =  ds_train_retina.__getitem__(int_id_retina)
        batched_retinas[i, :, :, :] = retina
    return batched_ecgs, batched_retinas

def batch_manager_eyes(ids_regCode, device, imsize, ds_train_retina_left = None, ds_train_retina_right = None):
    batched_retinas_left = torch.zeros([len(ids_regCode), 3, imsize, imsize], device = device)
    batched_retinas_right = torch.zeros([len(ids_regCode), 3, imsize, imsize], device = device)
    for i in range(min(len(ids_regCode), len(ids_regCode))):
        int_id_retina_left = ds_train_retina_left.fileset.index.get_loc(ids_regCode[i])
        retina_left =  ds_train_retina_left.__getitem__(int_id_retina_left)
        batched_retinas_left[i, :, :, :] = retina_left
        int_id_retina_right = ds_train_retina_right.fileset.index.get_loc(ids_regCode[i])
        retina_right =  ds_train_retina_right.__getitem__(int_id_retina_right)
        batched_retinas_right[i, :, :, :] = retina_right
    return batched_retinas_left, batched_retinas_right

def train_eyes_contrastive(steps,
                           pretrained_eyenet,
                           ds_train_retina_left,
                           ds_train_retina_right,
                           batch_size=10,
                           accumulation_size=1,
                           save_every=20,
                           lr1=1e-3,
                           lr3=0.001,
                           temp_0=0.3,
                           latent_shape = None,
                           orig_shape = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp = nn.Parameter(torch.ones(1, device = device)*temp_0, requires_grad=True)
    print("Using " + str(device))
    torch.cuda.empty_cache()
    people = ds_train_retina_left.ids ##we already made sure we are only training on people with both modalities
    people_batched_regCode = np.array_split(people, len(people)/batch_size)
    i = 0
    optimizer1 = optim.Adam(pretrained_eyenet.parameters(), lr=lr1)
    optimizer3 = optim.Adam([temp], lr=lr3)
    for epoch in range(10):
        for person_batch_regCode in people_batched_regCode:
            pretrained_eyenet.train()
            batched_eyes_raw_left, batched_eyes_raw_right = batch_manager_eyes(person_batch_regCode, device, 750, ds_train_retina_left = ds_train_retina_left, ds_train_retina_right = ds_train_retina_right)
            eye_embeddings_left = pretrained_eyenet(batched_eyes_raw_left, onlyEmbeddings = True, latent_shape = latent_shape, orig_shape = orig_shape)
            eye_embeddings_right = pretrained_eyenet(batched_eyes_raw_right, onlyEmbeddings = True, latent_shape = latent_shape, orig_shape = orig_shape)
            train_loss, reconst, contrast = combined_loss(eye_embeddings_left, eye_embeddings_right, None, None, None, None, temp, 0,0,0,0,0)
            train_loss.backward()
            if i % accumulation_size == 0:
                optimizer1.step()
                optimizer3.step()
                optimizer1.zero_grad()
                optimizer3.zero_grad()
                print("Contrast", contrast.item())
                if lr3 > 1e-8:
                    print("Temp trained to " + str(temp.item()))
            if i % save_every == 0 and i != 0:
                print("Saved checkpoint at i = " + str(i))
                torch.save(pretrained_eyenet, saved_model_path + "eye_contrastive_lr.pt")
            if i == steps:
                return pretrained_eyenet
            i += 1
        return pretrained_eyenet

def normalize(x):
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)

def train_together(steps,
                   epochs,
                      pretrained_eyenet,
                      pretrained_ecgnet,
                      ds_train_retina, train_loader_retina, ds_test_retina, test_loader_retina,
                      ds_train_ecg, train_loader_ecg, ds_test_ecg, test_loader_ecg, imsize,
                      batch_size = 10,
                    save_every = 20,
                      test_batch_size = 2,
                      vis_every = 10,
                      device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                      latent_shape = None,
                      orig_shape = None,
                      ecg = 0,
                      eye = 0,
                      star = 1,
                    lr = 1e-4):
    print("Using " + str(device))
    torch.cuda.empty_cache()
    people = ds_train_retina.ids ##we already made sure we are only training on people with both modalities
    people_batched_regCode = np.array_split(people, len(people)/batch_size)
    mapper = pd.read_csv(mapper_cache_loc).set_index("Unnamed: 0").to_dict()["0"]
    i = 1
    ##only decoder parameters of both networks
    optimizer = optim.Adam(list(pretrained_eyenet.parameters()) + list(pretrained_ecgnet.parameters()), lr = lr)
    for epoch in range(epochs):
        for person_batch_regCode in people_batched_regCode:
            pretrained_ecgnet.train()
            pretrained_eyenet.train()
            batched_ecgs_raw, batched_eyes_raw = batch_manager(person_batch_regCode, ds_train_ecg, ds_train_retina, mapper, device, imsize)
            eye_embeddings = pretrained_eyenet(batched_eyes_raw, onlyEmbeddings=True, latent_shape = latent_shape, orig_shape = orig_shape)
            ecg_embeddings = pretrained_ecgnet(batched_ecgs_raw, onlyEmbeddings=True)
            ecg_reconst = pretrained_ecgnet(onlyDecode=True, z = ecg_embeddings)
            eye_reconst = pretrained_eyenet(onlyDecode=True, z= eye_embeddings, latent_shape=latent_shape,orig_shape=orig_shape)
            train_loss = combined_loss(eye_embeddings, ecg_embeddings, eye_reconst, batched_eyes_raw, ecg_reconst, batched_ecgs_raw, alpha_ecg = ecg, alpha_eye=eye, star = star, device = device)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
           # print("train", train_loss.item())
            if i % save_every == 0:
                print("Saved checkpoint at i = " + str(i))
                torch.save(pretrained_ecgnet, saved_model_path + "ecg_contrastive.pt")
                torch.save(pretrained_eyenet, saved_model_path + "eye_contrastive.pt")
            if i % vis_every == 0:
                with torch.no_grad():
                    pretrained_ecgnet.eval()
                    pretrained_eyenet.eval()
                    for person_batch_regCode_test in np.array_split(ds_test_retina.ids, len(ds_test_retina.ids)/test_batch_size)[0:1]:
                        if len(person_batch_regCode_test) > 0:
                            batched_ecgs_raw, batched_eyes_raw = batch_manager(person_batch_regCode_test, ds_test_ecg,ds_test_retina, mapper, device, imsize)
                            int_id_retina = list(ds_test_retina.ids).index(person_batch_regCode_test[0])
                            eye_embeddings = pretrained_eyenet(batched_eyes_raw, onlyEmbeddings=True, latent_shape=latent_shape, orig_shape=orig_shape)
                            ecg_embeddings = pretrained_ecgnet(batched_ecgs_raw, onlyEmbeddings=True)
                            ecg_reconst = pretrained_ecgnet(onlyDecode=True, z=ecg_embeddings)
                            eye_reconst = pretrained_eyenet(onlyDecode=True, z=eye_embeddings,
                                                                latent_shape=latent_shape,
                                                                orig_shape=orig_shape)
                            compare(pd.DataFrame(ecg_reconst.detach().cpu()[0,:,:]), pd.DataFrame(batched_ecgs_raw[0,:,:].cpu()))
                            predicted = eye_reconst.detach().cpu()[0, :, :].permute(1, 2, 0)
                            plt.imshow(predicted)
                            plt.show()
                            actual_eye = ds_test_retina.__getitem__(int_id_retina)
                            plt.imshow(actual_eye.permute(1, 2, 0))
                            plt.show()

            if i == steps:
                return pretrained_eyenet, pretrained_ecgnet
            i += 1
    else:
        return pretrained_eyenet, pretrained_ecgnet


if __name__ == "__main__":
    np.random.seed(0)
    numTrain = 100
    numTest = 5
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #for debugging
    saved_model_path = "/net/mraid20/export/jasmine/zach/cross_modal/saved_models/"

    mapper_cache_loc = "/net/mraid20/export/jasmine/zach/cross_modal/ecg_mapper_cache.csv"
    all_ids_ecg_mapper = pd.read_csv(mapper_cache_loc).set_index("Unnamed: 0").to_dict()["0"]

    all_ids_retina = list(read_fileset().index.values)
    all_ids_ecg = list(all_ids_ecg_mapper.keys())
    all_ids_both = list(set(all_ids_ecg).intersection(all_ids_retina))

    train_ids_regCode = list(np.random.choice(all_ids_both, numTrain, replace=False))
    test_ids_regCode = list(set(all_ids_both) - set(train_ids_regCode))[0:numTest]

    train_ids_ecg = list(map(lambda x: all_ids_ecg_mapper.get(x), train_ids_regCode))
    test_ids_ecg = list(map(lambda x: all_ids_ecg_mapper.get(x), test_ids_regCode))

    assert len(set(train_ids_regCode).intersection(set(test_ids_regCode))) == 0
    print("generating dataloaders and datasets for fast loading")
    ds_train_retina_left_large, train_loader_retina_left_large, ds_test_retina_left_large, test_loader_retina_left_large = make_train_test_split_retina(
        train_ids_regCode, test_ids_regCode, 3000, "left")
    ds_train_retina_right_large, train_loader_retina_right_large, ds_test_retina_right_large, test_loader_retina_right_large = make_train_test_split_retina(
        train_ids_regCode, test_ids_regCode, 3000, "right")
    ds_train_retina_left, train_loader_retina_left, ds_test_retina_left, test_loader_retina_left = make_train_test_split_retina(
        train_ids_regCode, test_ids_regCode, 750, "left")
    ds_train_retina_right, train_loader_retina_right, _ds_test_retina_right, test_loader_retina_right = make_train_test_split_retina(train_ids_regCode, test_ids_regCode, 750, "right")
    ds_train_ecg, train_loader_ecg, ds_test_ecg, test_loader_ecg = make_train_test_split_ecg(train_ids_ecg, test_ids_ecg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device))
    if device == "cuda":
        torch.cuda.empty_cache()
    train_from_scratch_eye = False
    resave_eye = False
    if train_from_scratch_eye:
        ##train each autoencoder separately based on reconstruction loss
        eye_net = train_eye(1000, 1, eye().to(device), ds_train_retina_left_large, train_loader_retina_left_large,
                            ds_test_retina_left_large,
                            test_loader_retina_left_large, device, do_eval=False,
                            lr=4e-2,
                            latent_shape=(115, 74),
                            orig_shape=(3000, 3000))  # 5e-2 for the 3k image, 1e-2 for the 750 image

        if resave_eye:
            torch.save(eye_net, saved_model_path + "eye.pt")
        train_from_scratch_ecg = False
    resave_ecg = False
    if train_from_scratch_ecg:
        ecg_net = ecg().to(device)
        ecg_net = train_ecg(5000, 30,
                            ecg_net, ds_train_ecg, train_loader_ecg,
                            ds_test_ecg, test_loader_ecg,
                            device,
                            lr=5e-4,
                            alpha=0, beta=0, gamma=0,
                            delta=1, theta=1 / 400000,
                            do_eval=True)
        if resave_ecg:
            torch.save(ecg_net, saved_model_path + "ecg.pt")
    retrain_lr = False
    if retrain_lr:
        ##train left vs right eye contrastively

        train_eyes_contrastive(5,
                               torch.load(saved_model_path + "eye.pt"),
                               ds_train_retina_left = ds_train_retina_left,
                               ds_train_retina_right = ds_train_retina_right,
                               batch_size=20,
                               accumulation_size=1,
                               save_every=5,
                               lr1=2e-3,
                               lr3=0,
                               temp_0=0.2,
                               latent_shape=(46, 44),
                               orig_shape=(750, 750),
                               )
    ##need Quadro_rtx_6000
    #### very experimental
    torch.cuda.empty_cache()
    a,b= train_together(100, 1000,
                    torch.load(saved_model_path + "eye.pt"),
                    torch.load(saved_model_path + "ecg.pt"),
                    ds_train_retina_left_large, train_loader_retina_left_large, ds_test_retina_left_large, test_loader_retina_left_large,
                    ds_train_ecg, train_loader_ecg, ds_test_ecg, test_loader_ecg, 3000,
                    batch_size= 5,
                    save_every = 1000,
                    vis_every= 25,
                    device = device,
                    latent_shape=(46, 185),
                    orig_shape=(3000, 3000),
                    ecg = 1,
                    eye = 10,
                    star = 4, lr = 1e-3)
