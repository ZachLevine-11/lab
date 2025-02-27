import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

xgb_res_dl = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/disease_pred/xgboost/res_dl.csv")
xgb_res_hrv = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/disease_pred/xgboost/res_hrv.csv").loc[:, ["Unnamed: 0", "AUC_PRC", "AUC_ROC", "Cases Ratio"]]

mlp_res_dl= pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/disease_pred/mlp/res_dl.csv")
mlp_res_hrv = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/cross_modal/disease_pred/mlp/res_hrv.csv").loc[:, ["Unnamed: 0", "AUC_PRC", "AUC_ROC", "Cases Ratio"]]
xgb_res_hrv["Base_Data"] = "Clinical ECG Features"
mlp_res_hrv["Base_Data"] = "Clinical ECG Features"
mlp_res_dl["Base_Data"] = "SSL"
xgb_res_dl["Base_Data"] = "SSL"
mlp_res_dl["Model"] = "MLP"
mlp_res_dl["Model"] = "MLP"

xgb_res_dl["Model"] = "XGBoost"
xgb_res_hrv["Model"] = "XGBoost"
mlp_res_hrv["Model"] = "MLP"
stacked = pd.concat([xgb_res_dl, xgb_res_hrv, mlp_res_dl, mlp_res_dl], join="inner", ignore_index=True).set_index("Unnamed: 0")

def make_plots(stacked = stacked, on = "AUC_PRC"):
    stacked["Relative_AUC_PRC"] = stacked["AUC_PRC"]/stacked["Cases Ratio"]
    diseases_we_win = list(set(filter(lambda disease: stacked.loc[disease,].sort_values(ascending = False, by = on).loc[:,"Base_Data"][0].startswith("SSL"),  set(stacked.index.values))))
    diseases_batched = np.array_split(diseases_we_win, 5)
    for set_of_diseases in diseases_batched:
        stacked_disease_set_only = stacked.loc[set_of_diseases, :]
        sns.barplot(data = stacked_disease_set_only, x = "Unnamed: 0", y = on, hue = stacked_disease_set_only[["Base_Data", "Model"]].apply(tuple, axis=1))
        plt.title(on + " of downstream disease prediction")
        plt.xlabel("ICD Disease Code")
        plt.show()
    diseases_batched_lost = np.array_split(list(set(stacked.index.values) - set(diseases_we_win)), 4)
    for set_of_diseases in diseases_batched_lost:
        stacked_disease_set_only = stacked.loc[set_of_diseases, :]
        sns.barplot(data=stacked_disease_set_only, x="Unnamed: 0", y=on,
                    hue=stacked_disease_set_only[["Base_Data", "Model"]].apply(tuple, axis=1))
        plt.title(on + " of downstream disease prediction")
        plt.xlabel("ICD Disease Code")
        plt.show()
    stacked["Base_Data_High_Level"] = list(map(lambda x: "SSL" if x.startswith("SSL") else "ECG Clinical Features", stacked["Base_Data"]))
    order = mlp_res_hrv["Unnamed: 0"]
    b1 = mlp_res_dl.set_index("Unnamed: 0")[["AUC_PRC"]].loc[order]
    b2 = mlp_res_hrv.set_index("Unnamed: 0")[["AUC_PRC"]].loc[order]
    sns.scatterplot(x = b2.values.flatten(), y = b1.values.flatten())
    plt.plot([0, 0.6], [0, 0.6], color="black")
    plt.xlabel("Clinical")
    plt.ylabel("Embeddings")
    plt.title("Comparing PRAUC predictons")
    plt.show()

##sometimes the previous function breaks graphics generation in pycharm when it gets stuck. this will fix the hangup.
def clean_graphics_window():
    plt.close()
    plt.clf()
    plt.cla()
    plt.plot(1)
    plt.show()