##use the deep_learning conda env
import numpy as np
import pandas as pd
import lightgbm as lgb
import numpy as np
import statsmodels.stats.multitest
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from matplotlib import pyplot as plt
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader
import scipy
from scipy.stats import pearsonr
import statsmodels
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gseapy as gp
from gseapy import barplot

def make_diagnoses_df(base_data, id_list = list(range(100)) + list(range(1000, 1011, 1))):
    b = MedicalConditionLoader().get_data(study_ids=id_list).df.reset_index()
    counts = base_data.merge(b, left_index=True, right_on="RegistrationCode",  how="inner").medical_condition.value_counts()
    diseases = list(counts.index[counts > 20])
    ##not worth predicting:
    diseases.remove("6A05") ##ADHD
    diseases.remove("RA01") ##covid -not reliable
    ##remove depression
    diseases = list(filter(lambda x: not x.startswith("Block"), diseases))
    res = pd.DataFrame(np.zeros([len(list(set(base_data.index.get_level_values(0)))), len(diseases)]), dtype = int)
    res["RegistrationCode"] = list(set(base_data.index.get_level_values(0)))
    res = res.set_index(["RegistrationCode"])
    res.columns = diseases
    b = b.set_index("RegistrationCode").loc[list(set(base_data.index.get_level_values(0)).intersection(set(b["RegistrationCode"])))].reset_index()
    for cond_string in diseases:
        people_with_disease = list(set(b.loc[b.medical_condition == cond_string, :]["RegistrationCode"]))
        res.loc[people_with_disease, cond_string] = 1
    return res

def calculate_age_month(birthday, sample_day):
    print(birthday)
    return sample_day.year - birthday.year - ((sample_day.month) < (birthday.month))

def make_age_df(emb_df, metadata_df, resolution = "month"):
    emb_df = emb_df.merge(pd.to_datetime(metadata_df["sample_date"]), left_index = True, right_index = True)
    subs = SubjectLoader().get_data(study_ids=list(range(100)) + list(range(1000, 1011, 1))).df
    subs = subs.reset_index(["Date"], drop=True).dropna(subset=["yob", "month_of_birth"])
    subs_merged_raw = emb_df.merge(subs, left_index=True, right_index=True)
    age_computation = subs_merged_raw.loc[:, ["month_of_birth", "yob"]]
    age_computation.columns = ["month", "year"]
    age_computation["day"] = 15
    subs_merged_raw["birthdate"] = pd.to_datetime(age_computation)
    if resolution != "day":
        subs_merged_raw["sample_date"].dt.strftime('%Y-%m')
        subs_merged_raw["birthdate"].dt.strftime('%Y-%m')
        age_raw = list(map(lambda x: calculate_age_month(x[0], x[1]), zip(subs_merged_raw["birthdate"], subs_merged_raw["sample_date"])))
    else:
        age_raw = list(map(lambda x: (x[1] - x[0]).days, zip(subs_merged_raw["birthdate"], subs_merged_raw["sample_date"])))
    age_df_raw = pd.DataFrame(age_raw)
    age_df_raw["RegistrationCode"] = list(subs_merged_raw.index)
    age_df_raw = age_df_raw.rename({0: "age"}, axis=1)
    age_df_raw = age_df_raw.set_index(["RegistrationCode"])
    return age_df_raw

gene_sets = ["GO_Biological_Process_2023"]

def overrep(res):
    res = gp.enrichr(gene_list=list(c.loc[(c > 0.5).values,:].index.values),
                     gene_sets=gene_sets,
                     organism="Human",
                     outdir="/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/rna_descriptive/gseapy").results.sort_values(by="Overlap", ascending=False)
    res["Term"] = res["Term"].str.split(" \(GO").str[0]
    barplot(res,
            column="Adjusted P-value",
            size=20,
            top_term=20,
            color="blue",
            title="Genes with R^2 > 0.5 from CBC, age, gender")
    plt.title("Genes with R^2 > 0.5 from CBC, age, gender")
    plt.savefig("/home/zacharyl/Desktop/myplot.png", dpi = 400)
    res = res.loc[res["Adjusted P-value"] < 0.05, :].sort_values("Adjusted P-value")
    return res

if __name__ == "__main__":
    do_fig_1 = True
    if do_fig_1:
        subs = SubjectLoader().get_data().df.reset_index().set_index("RegistrationCode")["gender"]
        subs = subs.loc[~subs.index.duplicated()].dropna()
        greg = pd.read_pickle(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/rna_options_final/" + "after_batch_correction_filtered_1000_no_regress_5mln_sample.df")
        extra = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/metadata_with_techn.csv")
        extra["RegistrationCode"] = list(map(lambda x: "10K_" + str(x), extra.participant_id))
        extra.set_index("RegistrationCode", inplace=True)
        age = make_age_df(greg, extra)

        ##predict age from the top 1200 most variably expressed genes
        X = pd.read_pickle(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/rna_options_final/" + "after_batch_correction_filtered_1000_no_regress_5mln_sample.df")
        X = np.log1p(np.clip(X, -0.999, None, ))
        e = X.merge(age, left_index = True, right_index = True)
        e = e.loc[~e.index.duplicated(keep = "first"),:]
        g = e.merge(subs, left_index=True, right_index=True, how="inner")
        ##fig 1
        rs_men = {}
        rs_women = {}
        for var in g.columns:
            if var != "age" and var != "gender":
                rs_men[var] = tuple(scipy.stats.pearsonr(g.loc[g["gender"] == 1.0, var], g.loc[g["gender"] == 1.0, "age"]))
                rs_women[var] = tuple(
                    scipy.stats.pearsonr(g.loc[g["gender"] == 0.0, var], g.loc[g["gender"] == 0.0, "age"]))

        rs_men = pd.DataFrame(rs_men)
        rs_men.index = ["r", "P"]
        rs_women = pd.DataFrame(rs_women)
        rs_women.index = ["r", "P"]
        rs_men = rs_men.T
        rs_women = rs_women.T
        rs_men["gender"] = "men"
        rs_women["gender"] = "women"
        rs = pd.concat([rs_men, rs_women])
        rs = rs.loc[rs.index != "age", :]
        rs["P"] = statsmodels.stats.multitest.multipletests(rs["P"], method="bonferroni")[1]
        rs_sig = rs.loc[rs["P"] < 0.05, :]
        rs_sig_women = rs_sig.loc[rs_sig["gender"] == "women", :]
        rs_sig_men = rs_sig.loc[rs_sig["gender"] == "men", :]

        # Create the plot
        plt.figure(figsize=(20, 6))
        sns.barplot(
            data=rs_sig_women,
            x=rs_sig_women.index,
            y='r',
            alpha=0.8,
            errorbar='se'  # Show standard error bars
        )

        # Customize the plot
        plt.title("Significant (after bonferonni) Gene~age associations, women", fontsize=16, pad=20)
        plt.xlabel('Genes', fontsize=12)
        plt.ylabel('r', fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Show the plot
        plt.show()

        ##other option
        plt.figure(figsize=(8, 12))

        # sns.heatmap(corr2.dropna(how='all').dropna(axis=1,how='all'), cmap="RdBu_r",vmin=-0.25, vmax=0.25, )
        sns.heatmap(rs_sig.pivot(columns=["gender"]).fillna(0)["r"], cmap=sns.color_palette("RdBu_r", as_cmap=True)).set(
            title="Nastya asked me to make this")
        plt.show()
    do_another_plot = False
    if do_another_plot:
        def do_the_plot(fname):
            cool_fname = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/rna_descriptive/" + fname
            df = pd.read_csv(cool_fname).set_index("Unnamed: 0")
            df_corrected = pd.DataFrame(
                multipletests(df.dropna().to_numpy().flatten(), method="bonferroni")[1].reshape(df.dropna().shape))
            df_corrected.columns = list(map(lambda x: x.split("pvalue_")[-1], df.columns))
            df_corrected.index = df.dropna().index
            sns.clustermap(
                -np.log10(df_corrected.loc[(df_corrected < 5e-2).any(axis=1), (df_corrected < 5e-2).any(axis=0)]),
                cmap=sns.color_palette("RdBu_r", as_cmap=True))
            plt.title(fname.split(".csv")[0])
            plt.show()


        files = os.listdir("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/rna_descriptive/")
        for file in files:
            try:
                do_the_plot(file)
            except Exception:
                pass

        # Predict each gene from CBC counts + age + gender + BMI

    ##This paragraph here now, showing the prediction of each gene from CBC counts + age + gender + BMI.
    do_cbc_plot = True
    if do_cbc_plot:
        # cbc
        bt_df = BloodTestsLoader().get_data().df
        bt_df_meta = BloodTestsLoader().get_data().df_metadata
        cbc_lab_ind = bt_df_meta[bt_df_meta['kupat_holim'] == 'tenk-cbc'].index
        bt_df_cbc = bt_df.loc[cbc_lab_ind].dropna(how='all', axis=1)
        bt_df_cbc_age = bt_df_meta.loc[cbc_lab_ind]['age'].reset_index().set_index("RegistrationCode")
        bt_df_cbc = bt_df_cbc.reset_index(drop=False).set_index("RegistrationCode")
        bt_df_cbc_age = bt_df_cbc_age.loc[~bt_df_cbc_age.index.duplicated(keep="last"), :]
        bt_df_cbc = bt_df_cbc.loc[~bt_df_cbc.index.duplicated(keep="last"), :]
        b = bt_df_cbc.loc[~bt_df_cbc.index.duplicated(), :].dropna()

        subs = SubjectLoader().get_data(study_ids=list(range(100)) + list(range(1000, 1011, 1))).df
        subs = subs.reset_index(["Date"])
        subs = subs.loc[~subs.index.duplicated(keep="first"), :]
        gender_df = subs["gender"].dropna()
        body = BodyMeasuresLoader().get_data(study_ids=list(range(100)) + list(range(1000, 1011, 1))).df
        body = body.reset_index(["Date"]).sort_values(ascending=True, by="Date")
        body = body.loc[~body.index.duplicated(keep="first"), :]
        bmi_df = body["bmi"].dropna()
        gender_bmi_df = pd.DataFrame(gender_df).merge(bmi_df, left_index=True, right_index=True)

        ##rna
        greg = pd.read_pickle(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/rna_options_final/" + "after_batch_correction_filtered_1000_no_regress_5mln_sample.df")
        extra = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/metadata_with_techn.csv")
        extra["RegistrationCode"] = list(map(lambda x: "10K_" + str(x), extra.participant_id))
        extra.set_index("RegistrationCode", inplace=True)
        age = make_age_df(greg, extra)
        X = pd.read_pickle(
            "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/rna_options_final/" + "after_batch_correction_filtered_1000_no_regress_5mln_sample.df")
        X = np.log1p(np.clip(X, -0.999, None, ))

        df = X.merge(age, left_index=True, right_index=True).merge(b, left_index=True, right_index=True)
        df = df.merge(gender_bmi_df, left_index=True, right_index=True)

        df = df.loc[~df.index.duplicated(keep="first"), :]
        df = df.loc[:, list(filter(lambda x: x != "Date", df.columns))]
        i = 0
        res = {}
        predictors = list(b.columns)
        predictors.remove("Date")
        for gene in X.columns:
            print(i)
            little_X = df[predictors + ["age", "gender", "bmi"]].values
            little_y = df[gene].astype(float).values
            res[gene] = lgb.LGBMRegressor(max_depth=3, learning_rate=0.01, n_estimators=1500, verbose = 0).fit(little_X, little_y).score(little_X, little_y)
            i += 1
        pd.Series(res).to_csv("~/Desktop/res.csv")