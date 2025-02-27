import pandas as pd
import numpy as np
import torch

if __name__ == "__main__":
    do_kl_figure = False
    if do_kl_figure:
        ##does the kl divergence between the sampling distributions of two different people correlate to the age difference between them?
        ##most importantly, the model was not trained on age [though it was trained on things related to it]
        scale2 = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/dist/scale_one.csv").set_index(
            "Unnamed: 0")
        loc2 = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ppg/dist/loc_one.csv").set_index("Unnamed: 0")
        scale2.columns = list(map(lambda x: str(x) + "_sd", scale2.columns))
        loc2.columns = list(map(lambda x: str(x) + "_mean", loc2.columns))
        merged = loc2.merge(scale2, left_index=True, right_index=True)
        merged.index.name = "RegistrationCode"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kl_pairwise = {}
        counter = 0
        num_chunks = 3 * 13  ##needs to divide perfectly
        l1 = np.array_split(merged.index, num_chunks)
        l2 = np.array_split(merged.index, num_chunks)
        kl_pairwise = np.zeros((len(merged.index), len(merged.index), 25))
        for i, person_chunk1 in enumerate(l1):
            for j, person_chunk2 in enumerate(l2):
                ##this works through broadcasting
                kl_pairwise[(len(person_chunk1) * i):(len(person_chunk1) * (i + 1)),
                (len(person_chunk2) * j):(len(person_chunk2) * (j + 1)), :] = torch.distributions.kl.kl_divergence(
                    Normal(torch.from_numpy(merged.loc[person_chunk1, :].iloc[:, :25].values)[:, None, :].to(device),
                           torch.from_numpy(merged.loc[person_chunk1, :].iloc[:, 25:].values)[:, None, :].to(device)),
                    Normal(torch.from_numpy(merged.loc[person_chunk2, :].iloc[:, :25].values)[None, :, :].to(device),
                           torch.from_numpy(merged.loc[person_chunk2, :].iloc[:, 25:].values)[None, :, :].to(
                               device))).cpu().numpy()
                counter += len(person_chunk1) * len(person_chunk2)
                print(100 * counter / (len(merged.index) ** 2))

        same_samples = list(set(all_concat_mech_ecg.index).intersection(list(set(comparator.index))))
        comparator_here = (comparator.loc[same_samples] - comparator.loc[same_samples].mean()) / comparator.loc[
            same_samples].std()
        comparator_pca_1 = eval_clusters_medical_conditions(comparator_here, diseases).set_index("Label")
        comparator_pca_1["type"] = "comparator"

        p = pd.DataFrame(PCA(10).fit_transform(kl_pairwise.sum(axis=(-2))), merged.index).loc[same_samples]
        kl = eval_clusters_medical_conditions(p, diseases).set_index("Label").sort_values("AMI", ascending=False)
        kl["type"] = "KL"
        combined = pd.concat([kl, comparator_pca_1], axis=0).reset_index().set_index(["Label", "type"],
                                                                                     append=True).sort_values("AMI",
                                                                                                              ascending=False)
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            diseases_valid = combined.loc[combined["NMI"] > 0.01, :].index.get_level_values(1).unique()
            sns.barplot(data=combined.reset_index().set_index("Label").loc[diseases_valid], hue="type", x="Label",
                        y=combined.columns[i], ax=ax)
            ax.set_title(combined.columns[i])
            ax.set_xlabel("")
        plt.show()