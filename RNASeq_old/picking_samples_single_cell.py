import numpy as np
import pandas as pd
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader
from datetime import datetime, date
from matplotlib import pyplot as plt

##get the set of exclusion critera
base_dir = "/home/zacharyl/"
ex_med_df = pd.read_csv(base_dir + "Medications10KLoader_Healthy.csv")
ex_cond_df = pd.read_csv(base_dir + "medical_conditions_from_yotam.csv")
exclusion_medications = ex_med_df.loc[ex_med_df.loc[:, "Healthy Exclusion"] == 1,:]
exclusion_conds = ex_cond_df.loc[ex_cond_df.loc[:, "Healthy Group exclusion"] == 1,:]

##find people to exclude based on it
medications_df = Medications10KLoader().get_data().df.reset_index()
conditions_df = MedicalConditionLoader().get_data().df.reset_index()
people_to_excude_based_on_medications = medications_df.loc[list(map(lambda x: True if x in exclusion_medications.column_name.values else False, medications_df.medication.values)),:]
people_to_exclude_based_on_conditions = conditions_df.loc[list(map(lambda x: True if x in exclusion_conds.column_name.values else False, conditions_df.medical_condition.values)),:]
all_people_to_exclude = list(set(people_to_exclude_based_on_conditions.RegistrationCode).union(people_to_excude_based_on_medications.RegistrationCode))

##add age to the pbmc sample list
fname_all_samples = "/net/mraid08/export/genie/LabData/Data/10K/lab_collector/pbmc_tenk.csv"
df_all = pd.read_csv(fname_all_samples)
df_all = df_all.drop_duplicates(subset = "RegistrationCode", keep = "last")
df_all["Date_Sample"] = list(map(lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d").date(), df_all.returned_at))
subjects_all = SubjectLoader().get_data().df.reset_index().drop_duplicates(subset = "RegistrationCode", keep = "last")
subjects_all["DOB"] = 15

def calculate_age(born):
    today = date.today()
    return today.year - born["yob"] - ((today.month, today.day) < (born["month_of_birth"], born["DOB"]))
merged = subjects_all.merge(df_all, how = "inner", left_on = "RegistrationCode", right_on = "RegistrationCode")
merged = merged.dropna(subset = ["yob", "month_of_birth"])
merged["age"] = merged.apply(calculate_age, axis = 1)

pbmc_healthy = merged.loc[list(map(lambda x: x not in all_people_to_exclude, merged.RegistrationCode)),:]
pbmc_healthy = pbmc_healthy.loc[pbmc_healthy.age > 0, :]##exclude outliers and people with invalid  age
##keep only women
pbmc_healthy = pbmc_healthy.loc[pbmc_healthy.gender < 1, :] ##based on the metadata, 0: women, 1:men

age_range = pbmc_healthy.age.max()- pbmc_healthy.age.min()
bin_width = 2
age_bins = list(zip(list(range(int(pbmc_healthy.age.min()), int(pbmc_healthy.age.max()), bin_width + 1)), list(range(int(pbmc_healthy.age.min()) + bin_width, int(pbmc_healthy.age.max()), bin_width + 1))))
##if we didn't hit the end of the age range, include a manual last bin
if age_bins[-1][1] < pbmc_healthy.age.max():
    age_bins.append((age_bins[-1][1] + 1, int(pbmc_healthy.age.max())))

##pick all people at the tails of the age distribution
upper_bound_all_samples = 39
lower_bound_all_samples = 70
number_per_bin = 10
##for each middle bin interval, randomly pick the set number of people in that age bin
samples = list(map(lambda bin_interval: np.random.choice(pbmc_healthy.loc[list(map(lambda age: bin_interval[0] <= age <= bin_interval[1], pbmc_healthy.age)), :].RegistrationCode, size = min(number_per_bin, len(pbmc_healthy.loc[list(map(lambda age: bin_interval[0] <= age <= bin_interval[1], pbmc_healthy.age)), :].RegistrationCode)), replace = False) if len(set(list(range(bin_interval[0], bin_interval[1] + 1, 1))).intersection(set(pbmc_healthy.age))) > 0 else None, list(filter(lambda x: x[0] > upper_bound_all_samples and x[1] < lower_bound_all_samples, age_bins))))
##drop the None entries (indicating no person has that age bin) from the list
samples = list(filter(lambda x: x is not None, samples))
##unpack the sampled tuples for each age bin
samples = list(set([y for x in samples for y in x]))
##include all young people
samples += list(pbmc_healthy.loc[pbmc_healthy.age <= upper_bound_all_samples, "RegistrationCode"].unique())
##include all old people
samples += list(pbmc_healthy.loc[pbmc_healthy.age >= lower_bound_all_samples, "RegistrationCode"].unique())

##Add required metadata for each sample that we are picking
samples_full = pd.DataFrame({"RegistrationCode": samples})
samples_full = samples_full.merge(pbmc_healthy, on = "RegistrationCode", how = "inner")

##Make table with ages'
table = pd.DataFrame(age_bins)
table.columns = ["Lower_bound", "Upper_bound"]
number_per_bin = list(map(lambda bin_interval: len(pbmc_healthy.loc[list(map(lambda age: bin_interval[0] <= age <= bin_interval[1], pbmc_healthy.age)), :].RegistrationCode.unique()), age_bins))
table["number_per_bin"] =number_per_bin

##make a histogram of the age distrivution of the selected people
plt.hist(samples_full["age"], list(range(int(pbmc_healthy.age.min()), int(pbmc_healthy.age.max()), 2)))
plt.show()
