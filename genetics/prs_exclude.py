from LabData.DataLoaders.PRSLoader import PRSLoader
from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader
from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
from LabData.DataLoaders.FamilyMedicalConditionsLoader import FamilyMedicalConditionsLoader
from modified_tom_functions import getsigunique
import json

def EnglishSicknesstoICDCode(english_str, medData):
    lookup_dict = medData.df_columns_metadata.set_index("english_name").ICD11Code.to_dict()
    ##Match to any word in a key
    for englishname, icdcode in lookup_dict.items():
        if english_str in englishname:
            thecode = icdcode
            return thecode
    return None

def excludeBasedOnPersonalMedicalConditions(whattheyhave, medData):
    medicalConditions = medData.df.reset_index()
    thecode = EnglishSicknesstoICDCode(whattheyhave, medData)
    if thecode == None:
        return []
    else:
        ##We have a match
        return list(
            medicalConditions.loc[medicalConditions["medical_condition"] == thecode, "RegistrationCode"].unique())

def prsToExcluded_people(prsName, h_2_description_dict):
    prsmeaning = h_2_description_dict[prsName]
    if type(prsmeaning) == float:  ##then there's no matching meaning, because this is NA
        return []
    if "Non-cancer illness code, self-reported: " in prsmeaning or "Diagnoses - main ICD10" in prsmeaning or "problem" in prsmeaning:
        whattheyhave = prsmeaning.split(": ")[-1]
        medData = MedicalConditionLoader().get_data()
        return excludeBasedOnPersonalMedicalConditions(whattheyhave, medData)
    elif "Treatment/medication code" in prsmeaning or "Medication for " in prsmeaning:
        what = prsmeaning.split(": ")[-1]
        if "None" in what:
            return []
        else:
            meddata = Medications10KLoader().get_data().df.reset_index()
            meddata["medication_english_only"] = list(map(lambda name: name.split("/")[-1], meddata["medication"]))
            return list(meddata.loc[list(map(lambda medEntry: what in medEntry,
                                             meddata["medication_english_only"])), "RegistrationCode"].unique())
    elif "Illnesses of " in prsmeaning:
        family = FamilyMedicalConditionsLoader().get_data().df.reset_index()
        who = prsmeaning.split("Illnesses of ")[-1].split(":")[0]
        what_english = prsmeaning.split(": ")[-1]
        what_icd = EnglishSicknesstoICDCode(what_english, medData=MedicalConditionLoader().get_data())
        if what_icd == None:
            return []
        else:
            hasAnyRelativeWithIllness = family.loc[family.medical_condition == what_icd, :]
            hasExactRelativeWithIllness = hasAnyRelativeWithIllness.loc[hasAnyRelativeWithIllness.relative == who, :]
            return list(hasExactRelativeWithIllness.RegistrationCode.unique())
    else:
        return []


##For each PRS get a list of people to exclude based on having that trait already
##Return the result as a dictionary or PRS:Exclude indices list key, value pairs
def get_exclusion_map(use_cached=True,
                      cache="/net/mraid20/export/jasmine/zach/prs_associations/exclusionmap_cache.txt"):
    ignoreList = [2247_0]  ##This one is "No" for hearing difficulty/problems
    if use_cached == True:
        with open(cache) as file:
            data = file.read()
            return json.loads(data)
    else:
        res = {}
        h2_description_dict = PRSLoader().get_data().df_columns_metadata.reset_index().set_index(
            "phenotype_code").h2_description.to_dict()
        for prs in getsigunique():
            if prs not in ignoreList:
                print("Now onto PRS: ", prs, " meaning: ", h2_description_dict[prs])
                toExclude = prsToExcluded_people(prs, h2_description_dict)
                res[prs] = toExclude
        with open(cache, "w") as file:
            file.write(json.dumps(res))
        return res
