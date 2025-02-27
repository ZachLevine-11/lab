import pandas as pd
from LabData.DataLoaders.CGMLoader import CGMLoader
from LabData.DataLoaders.PRSLoader import PRSLoader
from LabData.DataLoaders.DietLoggingLoader import DietLoggingLoader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    prs = PRSLoader().get_data()
    prs.df_columns_metadata.loc[prs.df_columns_metadata.h2_description == "Body mass index (BMI)", :]
    cgm = CGMLoader().get_data(study_ids=["10K"]).df

    obese_people = prs.df.loc[prs.df["23104_irnt"] > prs.df["23104_irnt"].mean() + prs.df["23104_irnt"].std() * 3,:].index
    obese_people = pd.Series(list(obese_people))
    underweight_people = prs.df.loc[prs.df["23104_irnt"] < prs.df["23104_irnt"].mean() - prs.df["23104_irnt"].std() * 3,:].index
    underweight_people = pd.Series(list(underweight_people))
    obese_people.name = "RegistrationCode"
    underweight_people.name = "RegistrationCode"

    ##compare the HHMM mean day vector for genetically overweight and underweight people
    cgm_ov = pd.merge(cgm.reset_index().set_index("RegistrationCode"), obese_people, left_index=True,
                      right_on="RegistrationCode", how="inner")
    cgm_uw = pd.merge(cgm.reset_index().set_index("RegistrationCode"), underweight_people, left_index=True,
                      right_on="RegistrationCode", how="inner")
    for i in range(min(len(obese_people), len(underweight_people))):
        cgm_ov_one_person = cgm_ov.loc[cgm_ov.RegistrationCode == obese_people[i], :]
        ow_ts_one_person = cgm_ov_one_person.groupby(cgm_ov_one_person.Date.dt.time).mean()
        cgm_uw_one_person = cgm_uw.loc[cgm_uw.RegistrationCode == underweight_people[i], :]
        uw_ts_one_person = cgm_uw_one_person.groupby(cgm_uw_one_person.Date.dt.time).mean()
        plt.plot(range(len(uw_ts_one_person)), uw_ts_one_person.GlucoseValue, color = "red", label = "uw")
        plt.plot(range(len(ow_ts_one_person)), ow_ts_one_person.GlucoseValue, color = "blue", label ="ow")
        plt.legend()
        plt.show()

    ##compare the eating patterns of genetically overweight and underweight people
    diet_ov = pd.merge(cgm.reset_index().set_index("RegistrationCode"), obese_people, left_index=True,
                      right_on="RegistrationCode", how="inner")
    diet_uw = pd.merge(cgm.reset_index().set_index("RegistrationCode"), underweight_people, left_index=True,
                      right_on="RegistrationCode", how="inner")
    for i in range(min(len(obese_people), len(underweight_people))):
        cgm_ov_one_person = cgm_ov.loc[cgm_ov.RegistrationCode == obese_people[i], :]
        ow_ts_one_person = cgm_ov_one_person.groupby(cgm_ov_one_person.Date.dt.time).mean()
        cgm_uw_one_person = cgm_uw.loc[cgm_uw.RegistrationCode == underweight_people[i], :]
        uw_ts_one_person = cgm_uw_one_person.groupby(cgm_uw_one_person.Date.dt.time).mean()
        plt.plot(range(len(uw_ts_one_person)), uw_ts_one_person.GlucoseValue, color = "red", label = "uw")
        plt.plot(range(len(ow_ts_one_person)), ow_ts_one_person.GlucoseValue, color = "blue", label ="ow")
        plt.legend()
        plt.show()