from GeneticsPipeline.config import gencove_logs_path
from LabData.DataLoaders.PRSLoader import PRSLoader
from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
import pandas as pd
import numpy as np
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from sklearn.linear_model import LinearRegression
import os
from LabQueue.qp import fakeqp as qp
from LabUtils.addloglevels import sethandlers


##fit a model to non missing data and subtract the predictons from the data you fit to
##in order to fit, x and y have to be alligned
##in order to
def correct_single_covariate(y, age, gender):
    if y.shape[1] > 1:  ##sometimes there are extra null columns with the same name
        y = pd.DataFrame(
            y.iloc[:, 0])  ##making this a dataframe lets dtypes below return an array and not a single value
    os.chdir(gencove_logs_path)
    y_corrected = y
    predicted_y = np.zeros_like(y)
    # predict metabolites from age and gender and subtract that from the values of original metabolites.
    ##but skip non numeric columns
    if y.dtypes[0] != "float64":  ##index at zero because technically the dtypes operator returns a series
        return y
    else:
        pass
    if len(age) != 0 and len(gender) != 0 and len(y) != 0 and len(y.dropna()) != 0:
        inputx = pd.DataFrame(age.reset_index()["age"], gender.reset_index()["gender"])
        model = LinearRegression().fit(inputx[np.logical_and(~y.isna(), np.array(~inputx.isna())).values],
                                       y[np.logical_and(~y.isna(), np.array(~inputx.isna())).values])
        predicted_y[np.logical_and(~y.isna(), np.array(~inputx.isna())).values] = model.predict(
            inputx.iloc[np.logical_and(~y.isna(), np.array(~inputx.isna())).values,]).flatten()
        # without_a_g is the value of the metabolites without age and gender
        y_corrected -= predicted_y
        ##return original metabolite
    return y_corrected


def correct_all_covariates(loader, cluster=True, use_imputed_version=True, index_is_10k=False, get_data_args=None,
                           use_precomputed_loader=False, precomputed_loader=None):
    pd.options.mode.use_inf_as_na = True  ##treat inf as na values to save number of checks
    os.chdir(gencove_logs_path)
    # sethandlers()
    with qp(q=['himem7.q'], jobname='clac_metabol', max_u=100, max_r=100, _suppress_handlers_warning=True) as q:
        ##to speed up, do before doing anyting else
        q.startpermanentrun()
        if not use_precomputed_loader:
            if loader == SerumMetabolomicsLoader:
                if cluster and not use_imputed_version:
                    y = SerumMetabolomicsLoader().get_data(
                        precomputed_loader_fname="metab_10k_data_RT_clustering").df.copy()
                if cluster and use_imputed_version:
                    y = pd.read_csv("/net/mraid20/export/jafar/Tom/imputed_cluster_indcol.csv")
            else:  ## Use a different loader
                if get_data_args is None:
                    y = loader().get_data(study_ids=['10K']).df.copy()
                else:
                    y = loader().get_data(get_data_args, study_ids=['10K']).df.copy()
                ##log correction
                y = y.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype,
                                                                   np.number) else x)  # replacing infinite value with NaN
                y.replace([np.inf, -np.inf], np.nan,
                          inplace=True)  ##doesn't really matter though because we already treat thse as the same thing in the pd options
                # removing the columns with more than 50% Null
                y = y.dropna(thresh=len(y) / 2, axis=1)
            if loader == SerumMetabolomicsLoader:
                if not use_imputed_version:  # imputed version already has the proper 10k index
                    y['RegistrationCode'] = list(map(lambda serum: '10K_' + serum.split('_')[0], y.index.values))
            else:  ##for other loaders
                if not index_is_10k:
                    y['RegistrationCode'] = list(map(lambda serum: '10K_' + serum.split('_')[0], y.index.values))
                else:
                    ##still drop the index because the preprocess_loader_data skips this step
                    y = y.reset_index(drop=False)
            y.set_index('RegistrationCode', inplace=True, drop=True)  # set this column to be the index of y
        else:
            y = precomputed_loader
        metadata = SubjectLoader().get_data()
        ga_metabdat = metadata.df.reset_index(level=-1)[['gender', 'age', 'yob']].copy()
        ##merge with age and gender
        y = pd.merge(ga_metabdat, y, how='right', right_on='RegistrationCode', left_on='RegistrationCode')
        # use linear regression in order to predict each metabolites value from age + gender
        res = {}
        numCols = len(y.columns[3:])
        i = 1
        for c in y.columns[3:]:
            print("Correcting: " + str(c) + ", " + str(i) + "/" + str(numCols))
            # predict the covariance of age and gender on the metabolitesl
            # output: metabolites table with the value after removing the covariance of age and gender
            res[c] = q.method(correct_single_covariate, (y[[c]], y[["age"]], y[["gender"]]))
            i += 1
        wait_ony = {c: q.waitforresult(v) for c, v in res.items()}
    theRes = pd.concat(wait_ony.values(), axis=1)  ##we don't care aout age and gender
    return theRes


def save(result, filename):
    result.to_csv("/net/mraid20/export/jasmine/zach/prs_associations/" + filename)


def getnotrawPrses():
    theprses = PRSLoader().get_data().df.columns
    theprses = list(set([x for x in theprses if not str(x).endswith("_raw")]))
    return theprses


def getsigunique(cached=True, dir="/net/mraid20/export/jasmine/zach/prs_associations/",
                 fileName="getsigunique_cache.csv"):
    # Load all the PRS from 10K
    if cached:  ##Much faster
        cached_list = list(pd.read_csv(dir + fileName).iloc[:,
                           1])  ##column 0 holds an aribitrary index, column 1 has the list we care about
        return cached_list
    mydata = PRSLoader().get_data().df.columns
    metadata_table = pd.read_excel(
        '/net/mraid20/export/genie/LabData/Data/10K/genetics/PRSice/SummaryStatistics/Nealelab/v3/UKBB_GWAS_Imputed_v3-File_Manifest_Release_20180731.xlsx',
        sheet_name='Manifest 201807', engine='openpyxl')
    metadata_table = metadata_table.loc[metadata_table['Phenotype Code'].notnull()]
    metadata_table = metadata_table.loc[metadata_table['Sex'].eq('both_sexes')]
    metadata_table.set_index('Phenotype Code', inplace=True)
    # metadata of the PRS that are related to genetics
    high_sig = mydata.df_columns_metadata[mydata.df_columns_metadata['h2_h2_sig'].eq('z7')]
    high_sig = high_sig.merge(metadata_table, left_index=True, right_index=True, how='inner')
    # dividing the PRS into groups according to their descriptions
    prefix_group = {'Illnesses of siblings': 'Family history',
                    'Illnesses of mother': 'Family history',
                    'Illnesses of father': 'Family history',
                    'Non-cancer illness code, self-reported': 'Medical conditions',
                    'Treatment/medication code': 'Medication',
                    'Diagnoses - main ICD10': 'Summary Diagnoses',
                    'Pain type(s) experienced in last month': 'Pain',
                    'Leisure/social activities': 'Social Support',
                    'Types of physical activity in last 4 weeks': 'Physical activity',
                    'Qualifications': 'Education',
                    'Medication for pain relief, constipation, heartburn': 'Medication',
                    'Types of transport used (excluding work)': 'Physical activity',
                    'Reason for glasses/contact lenses': 'Eyesight',
                    'How are people in household related to participant': 'Household',
                    'Mouth/teeth dental problems': 'Mouth',
                    'Destinations on discharge from hospital (recoded)': 'Summary Administration',
                    'Vascular/heart problems diagnosed by doctor': 'Medical conditions',
                    'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones': 'Medication',
                    'Smoking status': 'Smoking',
                    'Mineral and other dietary supplements': 'Medication',
                    'Medication for cholesterol, blood pressure or diabetes': 'Medication',
                    'Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor': 'Medical conditions',
                    'Hair/balding pattern': 'Male-specific factors',
                    'Illness, injury, bereavement, stress in last 2 years': 'Mental health',
                    'Spread type': 'Diet',
                    'Major dietary changes in the last 5 years': 'Diet',
                    'Never eat eggs, dairy, wheat, sugar': 'Diet',
                    'Milk type used': 'Diet',
                    'Tobacco smoking': 'Smoking',
                    'Bread type': 'Diet',
                    'Cereal type': 'Diet',
                    'Coffee type': 'Diet',
                    'Attendance/disability/mobility allowance': 'Other sociodemographic factors',
                    'Hearing difficulty/problems': 'Hearing', 'Vitamin and mineral supplements': 'Medication',
                    'Body mass index (BMI)': 'Body size measures',
                    'Weight': 'Body size measures',
                    'Own or rent accommodation lived in': 'Household',
                    'High light scatter reticulocyte percentage': 'Blood count',
                    'IGF-1 (quantile)': 'Blood biochemistry',
                    'Average total household income before tax': 'Household',
                    'Time spend outdoors in summer': 'Sun exposure',
                    'Time spent outdoors in winter': 'Sun exposure',
                    'Time spent watching television (TV)': 'Physical activity',
                    'Time spent using computer': 'Physical activity',
                    'Time spent driving': 'Physical activity',
                    'Duration of light DIY': 'Physical activity',
                    'Pulse rate, automated reading': 'Blood pressure',
                    'Cooked vegetable intake': 'Diet',
                    'Salad / raw vegetable intake': 'Diet',
                    'Fresh fruit intake': 'Diet',
                    'Dried fruit intake': 'Diet',
                    'Oily fish intake': 'Diet',
                    'Non-oily fish intake': 'Diet',
                    'Processed meat intake': 'Diet',
                    'Poultry intake': 'Diet',
                    'Beef intake': 'Diet',
                    'Pork intake': 'Diet',
                    'Cheese intake': 'Diet',
                    'Bread intake': 'Diet',
                    'Cereal intake': 'Diet',
                    'Lamb/mutton intake': 'Diet',
                    'Salt added to food': 'Diet',
                    'Tea intake': 'Diet',
                    'Coffee intake': 'Diet',
                    'Water intake': 'Diet',
                    'Variation in diet': 'Diet',
                    'Current tobacco smoking': 'Smoking',
                    'Past tobacco smoking': 'Smoking',
                    'Exposure to tobacco smoke outside home': 'Smoking',
                    'Average weekly red wine intake': 'Alcohol',
                    'Average weekly champagne plus white wine intake': 'Alcohol',
                    'Average weekly spirits intake': 'Alcohol',
                    'Average weekly beer plus cider intake': 'Alcohol',
                    'Alcohol intake frequency.': 'Alcohol',
                    'Frequency of friend/family visits': 'Social support',
                    'Length of mobile phone use': 'Electronic device use',
                    'Drive faster than motorway speed limit': 'Physical activity',
                    'Weekly usage of mobile phone in last 3 months': 'Electronic device use',
                    'Invitation to complete online 24-hour recall dietary questionnaire, acceptance': 'Diet by 24-hour recall',
                    'Sleep duration': 'Sleep',
                    'Getting up in morning': 'Sleep',
                    'Morning/evening person (chronotype)': 'Sleep',
                    'Nap during day': 'Sleep',
                    'Sleeplessness / insomnia': 'Sleep',
                    'Snoring': 'Sleep',
                    'Daytime dozing / sleeping (narcolepsy)': 'Sleep',
                    'Number of self-reported non-cancer illnesses': 'Medical conditions',
                    'Number of operations, self-reported': 'Operations',
                    'Number of treatments/medications taken': 'Medication',
                    'Hot drink temperature': 'Diet',
                    'Alcohol usually taken with meals': 'Alcohol',
                    'Alcohol intake versus 10 years previously': 'Alcohol',
                    'Comparative body size at age 10': 'Early life factors',
                    'Comparative height size at age 10': 'Early life factors',
                    'Facial ageing': 'Sun exposure',
                    'Maternal smoking around birth': 'Early life factors',
                    'Father\'s age at death': 'Family history',
                    'Townsend deprivation index at recruitment': 'Baseline characteristics',
                    'Mood swings': 'Mental health',
                    'Miserableness': 'Mental health',
                    'Irritability': 'Mental health',
                    'Sensitivity / hurt feelings': 'Mental health',
                    'Fed-up feelings': 'Mental health',
                    'Nervous feelings': 'Mental health',
                    'Worrier / anxious feelings': 'Mental health',
                    "Tense / 'highly strung'": 'Mental health',
                    'Worry too long after embarrassment': 'Mental health',
                    'Sitting height': 'Body size measures',
                    'Fluid intelligence score': 'Fluid intelligence / reasoning',
                    'Birth weight': 'Early life factors',
                    'Mean time to correctly identify matches': 'Reaction time',
                    "Suffer from 'nerves'": 'Mental health',
                    'Alcohol drinker status': 'Alcohol', 'Neuroticism score': 'Mental health',
                    'Number of fluid intelligence questions attempted within time limit': 'Fluid intelligence / reasoning',
                    'Forced expiratory volume in 1-second (FEV1), Best measure': 'Spirometry',
                    'Forced vital capacity (FVC), Best measure': 'Spirometry',
                    'Forced expiratory volume in 1-second (FEV1), predicted': 'Spirometry',
                    'Forced expiratory volume in 1-second (FEV1), predicted percentage': 'Spirometry',
                    'Ever smoked': 'Smoking', 'Loneliness, isolation': 'Mental health',
                    'Guilty feelings': 'Mental health',
                    'Risk taking': 'Mental health', 'Frequency of drinking alcohol': 'Alcohol',
                    'Frequency of consuming six or more units of alcohol': 'Alcohol',
                    'Ever felt worried, tense, or anxious for most of a month or longer': 'Mental health',
                    'Ever had prolonged loss of interest in normal activities': 'Mental health',
                    'General happiness': 'Mental health',
                    'Ever had prolonged feelings of sadness or depression': 'Mental health',
                    'Ever taken cannabis': 'Cannabis use',
                    'General happiness with own health': 'Mental health',
                    'Belief that own life is meaningful': 'Mental health',
                    'Ever thought that life not worth living': 'Mental health',
                    'Felt hated by family member as a child': 'Mental health',
                    'Physically abused by family as a child': 'Mental health',
                    'Felt very upset when reminded of stressful experience in past month': 'Mental health',
                    'Felt loved as a child': 'Mental health',
                    'Frequency of depressed mood in last 2 weeks': 'Mental health',
                    'Ever sought or received professional help for mental distress': 'Mental health',
                    'Ever suffered mental distress preventing usual activities': 'Mental health',
                    'Ever had period extreme irritability': 'Mental health',
                    'Frequency of unenthusiasm / disinterest in last 2 weeks': 'Mental health',
                    'Frequency of tenseness / restlessness in last 2 weeks': 'Mental health',
                    'Frequency of tiredness / lethargy in last 2 weeks': 'Mental health',
                    'Seen doctor (GP) for nerves, anxiety, tension or depression': 'Mental health',
                    'Seen a psychiatrist for nerves, anxiety, tension or depression': 'Mental health',
                    'Trouble falling or staying asleep, or sleeping too much': 'Depression',
                    'Recent feelings of tiredness or low energy': 'Depression',
                    'Substances taken for depression': 'Depression',
                    'Activities undertaken to treat depression': 'Depression',
                    'Able to confide': 'Social support',
                    'Answered sexual history questions': 'Sexual factors',
                    'Age first had sexual intercourse': 'Sexual factors',
                    'Lifetime number of sexual partners': 'Sexual factors',
                    'Overall health rating': 'General health', 'Impedance of whole body': 'Body composition',
                    'Impedance of leg (right)': 'Body composition',
                    'Impedance of leg (left)': 'Body composition', 'Impedance of arm (right)': 'Body composition',
                    'Impedance of arm (left)': 'Body composition',
                    'Leg fat percentage (right)': 'Body composition', 'Leg fat mass (right)': 'Body composition',
                    'Leg fat-free mass (right)': 'Body composition', 'Leg predicted mass (right)': 'Body composition',
                    'Leg fat percentage (left)': 'Body composition', 'Leg fat mass (left)': 'Body composition',
                    'Leg fat-free mass (left)': 'Body composition',
                    'Leg predicted mass (left)': 'Body composition', 'Arm fat percentage (right)': 'Body composition',
                    'Arm fat mass (right)': 'Body composition',
                    'Arm fat-free mass (right)': 'Body composition',
                    'Arm predicted mass (right)': 'Body composition',
                    'Arm fat percentage (left)': 'Body composition',
                    'Arm fat mass (left)': 'Body composition',
                    'Arm fat-free mass (left)': 'Body composition',
                    'Arm predicted mass (left)': 'Body composition',
                    'Trunk fat percentage': 'Body composition',
                    'Trunk fat mass': 'Body composition',
                    'Trunk fat-free mass': 'Body composition',
                    'Trunk predicted mass': 'Body composition',
                    'Doctor diagnosed hayfever or allergic rhinitis': 'Medical information',
                    'Age started wearing glasses or contact lenses': 'Eyesight',
                    'Long-standing illness, disability or infirmity': 'General health',
                    'Plays computer games': 'Electronic device use',
                    'Year ended full time education': 'Work environment',
                    'Hearing difficulty/problems with background noise': 'Hearing',
                    'Home location - north co-ordinate (rounded)': 'Home locations',
                    'Body fat percentage': 'Body composition',
                    'Whole body fat mass': 'Body composition',
                    'Whole body fat-free mass': 'Body composition',
                    'Whole body water mass': 'Body composition',
                    'Falls in the last year': 'General health',
                    'Basal metabolic rate': 'Body composition',
                    'Wheeze or whistling in the chest in last year': 'Breathing',
                    'Chest pain or discomfort': 'Chest pain',
                    'Relative age of first facial hair': 'Male-specific factors',
                    'Relative age voice broke': 'Male-specific factors',
                    'Diabetes diagnosed by doctor': 'Medical conditions',
                    'Fractured/broken bones in last 5 years': 'Medical conditions',
                    'Other serious medical condition/disability diagnosed by doctor': 'Medical conditions',
                    'Taking other prescription medications': 'Medication',
                    'Light smokers, at least 100 smokes in lifetime': 'Smoking',
                    'Reason for reducing amount of alcohol drunk': 'Alcohol',
                    'Age when periods started (menarche)': 'Female-specific factors',
                    'Birth weight of first child': 'Female-specific factors',
                    'Age at first live birth': 'Female-specific factors',
                    'Age at last live birth': 'Female-specific factors',
                    'Age started oral contraceptive pill': 'Female-specific factors',
                    'Ever used hormone-replacement therapy (HRT)': 'Female-specific factors',
                    'Age started smoking in former smokers': 'Smoking',
                    'Age high blood pressure diagnosed': 'Medical conditions',
                    'White blood cell (leukocyte) count': 'Blood count',
                    'Red blood cell (erythrocyte) count': 'Blood count',
                    'Haemoglobin concentration': 'Blood count',
                    'Haematocrit percentage': 'Blood count',
                    'Mean corpuscular volume': 'Blood count',
                    'Mean corpuscular haemoglobin': 'Blood count',
                    'Mean corpuscular haemoglobin concentration': 'Blood count',
                    'Red blood cell (erythrocyte) distribution width': 'Blood count',
                    'Platelet count': 'Blood count',
                    'Platelet crit': 'Blood count',
                    'Mean platelet (thrombocyte) volume': 'Blood count',
                    'Lymphocyte count': 'Blood count',
                    'Monocyte count': 'Blood count',
                    'Neutrophill count': 'Blood count',
                    'Eosinophill count': 'Blood count',
                    'Lymphocyte percentage': 'Blood count',
                    'Neutrophill percentage': 'Blood count',
                    'Eosinophill percentage': 'Blood count',
                    'Reticulocyte percentage': 'Blood count',
                    'Reticulocyte count': 'Blood count',
                    'Mean reticulocyte volume': 'Blood count',
                    'Mean sphered cell volume': 'Blood count',
                    'Immature reticulocyte fraction': 'Blood count',
                    'High light scatter reticulocyte count': 'Blood count',
                    'Creatinine (enzymatic) in urine': 'Urine assays',
                    'Potassium in urine': 'Urine assays',
                    'Sodium in urine': 'Urine assays',
                    '3mm weak meridian (left)': 'Autorefraction',
                    '6mm weak meridian (left)': 'Autorefraction',
                    '6mm weak meridian (right)': 'Autorefraction',
                    '3mm weak meridian (right)': 'Autorefraction',
                    '3mm strong meridian (right)': 'Autorefraction',
                    '6mm strong meridian (right)': 'Autorefraction',
                    '6mm strong meridian (left)': 'Autorefraction',
                    '3mm strong meridian (left)': 'Autorefraction',
                    '3mm strong meridian angle (left)': 'Autorefraction',
                    'Standing height': 'Body measures',
                    'Hip circumference': 'Body measures',
                    'Albumin (quantile)': 'Blood biochemistry',
                    'Alanine aminotransferase (quantile)': 'Blood biochemistry',
                    'Aspartate aminotransferase (quantile)': 'Blood biochemistry',
                    'Urea (quantile)': 'Blood biochemistry',
                    'Calcium (quantile)': 'Blood biochemistry',
                    'Creatinine (quantile)': 'Blood biochemistry',
                    'Gamma glutamyltransferase (quantile)': 'Blood biochemistry',
                    'Glycated haemoglobin (quantile)': 'Blood biochemistry',
                    'Phosphate (quantile)': 'Blood biochemistry',
                    'Testosterone (quantile)': 'Blood biochemistry',
                    'Total protein (quantile)': 'Blood biochemistry',
                    'Forced vital capacity (FVC)': 'Spirometry',
                    'Forced expiratory volume in 1-second (FEV1)': 'Spirometry',
                    'Peak expiratory flow (PEF)': 'Spirometry',
                    'Ankle spacing width': 'Bone-densitometry of heel',
                    'Heel Broadband ultrasound attenuation, direct entry': 'Bone-densitometry of heel',
                    'Heel quantitative ultrasound index (QUI), direct entry': 'Bone-densitometry of heel',
                    'Heel bone mineral density (BMD)': 'Bone-densitometry of heel',
                    'Age at menopause (last menstrual period)': 'Female-specific factors',
                    'Number of incorrect matches in round': 'Pairs matching',
                    'Time to complete round': 'Pairs matching',
                    'Duration to first press of snap-button in each round': 'Reaction time',
                    'Diastolic blood pressure, automated reading': 'Blood pressure',
                    'Systolic blood pressure, automated reading': 'Blood pressure',
                    'Ankle spacing width (left)': 'Bone-densitometry of heel',
                    'Heel broadband ultrasound attenuation (left)': 'Bone-densitometry of heel',
                    'Heel quantitative ultrasound index (QUI), direct entry (left)': 'Bone-densitometry of heel',
                    'Heel bone mineral density (BMD) (left)': 'Bone-densitometry of heel',
                    'Heel bone mineral density (BMD) T-score, automated (left)': 'Bone-densitometry of heel',
                    'Ankle spacing width (right)': 'Bone-densitometry of heel',
                    'Heel broadband ultrasound attenuation (right)': 'Bone-densitometry of heel',
                    'Hospital episode type': 'Summary administration',
                    'Spells in hospital': 'Summary administration',
                    'Heel quantitative ultrasound index (QUI), direct entry (right)': 'Bone-densitometry of heel',
                    'Heel bone mineral density (BMD) (right)': 'Bone-densitometry of heel',
                    'Heel bone mineral density (BMD) T-score, automated (right)': 'Bone-densitometry of heel',
                    'Pulse rate': 'Arterial stiffness',
                    'Pulse wave reflection index': 'Arterial stiffness',
                    'Duration screen displayed': 'Prospective memory',
                    'Happiness': 'Mental health',
                    'Health satisfaction': 'Mental health',
                    'Family relationship satisfaction': 'Mental health',
                    'Friendships satisfaction': 'Mental health',
                    'Financial situation satisfaction': 'Mental health',
                    'Ever depressed for a whole week': 'Mental health',
                    'Hand grip strength (left)': 'Hand grip strength',
                    'Leg pain on walking': 'Claudication and peripheral artery disease',
                    'Hand grip strength (right)': 'Hand grip strength',
                    'Tinnitus': 'Hearing',
                    'Noisy workplace': 'Hearing',
                    'Waist circumference': 'Body size measures',
                    'FI3 ': 'Fluid intelligence / reasoning',
                    'FI4 ': 'Fluid intelligence / reasoning',
                    'FI6 ': 'Fluid intelligence / reasoning',
                    'FI8 ': 'Fluid intelligence / reasoning',
                    'Spherical power (right)': 'Autorefraction',
                    'Spherical power (left)': 'Autorefraction',
                    '3mm cylindrical power (right)': 'Autorefraction',
                    '3mm cylindrical power (left)': 'Autorefraction',
                    'Intra-ocular pressure, corneal-compensated (right)': 'Intraocular pressure',
                    'Intra-ocular pressure, Goldmann-correlated (right)': 'Intraocular pressure',
                    'Corneal hysteresis (right)': 'Intraocular pressure',
                    'Corneal resistance factor (right)': 'Intraocular pressure',
                    'Intra-ocular pressure, corneal-compensated (left)': 'Intraocular pressure',
                    'Intra-ocular pressure, Goldmann-correlated (left)': 'Intraocular pressure',
                    'Corneal hysteresis (left)': 'Intraocular pressure',
                    'Corneal resistance factor (left)': 'Intraocular pressure',
                    'Gas or solid-fuel cooking/heating': 'Household',
                    'Current employment status': 'Employment',
                    'Length of time at current address': 'Household',
                    'Number of vehicles in household': 'Household',
                    'Heel bone mineral density (BMD) T-score, automated': 'Bone-densitometry of heel',
                    'Job involves mainly walking or standing': 'Employment',
                    'Job involves heavy manual or physical work': 'Employment',
                    'Job involves shift work': 'Employment',
                    'Age completed full time education': 'Education',
                    'Number of days/week walked 10+ minutes': 'Physical activity',
                    'Duration of walks': 'Physical activity',
                    'Number of days/week of moderate physical activity 10+ minutes': 'Physical activity',
                    'Duration of moderate activity': 'Physical activity',
                    'Number of days/week of vigorous physical activity 10+ minutes': 'Physical activity',
                    'Duration of vigorous activity': 'Physical activity',
                    'Usual walking pace': 'Physical activity',
                    'Frequency of stair climbing in last 4 weeks': 'Physical activity',
                    'Frequency of walking for pleasure in last 4 weeks': 'Physical activity',
                    'Carpal tunnel syndrome': 'Nervous system disorders',
                    'Nerve, nerve root and plexus disorders': 'Nervous system disorders',
                    'Disorders of lens': 'Eyesight',
                    'Major coronary heart disease event': 'Circulatory system disorders',
                    'Coronary atherosclerosis': 'Circulatory system disorders',
                    'Diseases of veins, lymphatic vessels and lymph nodes, not elsewhere classified': 'Circulatory system disorders',
                    'Ischaemic heart disease, wide definition': 'Circulatory system disorders',
                    'Any ICDMAIN event in hilmo or causes of death': 'Circulatory system disorders',
                    'Diseases of the circulatory system': 'Circulatory system disorders',
                    'Disorders of gallbladder, biliary tract and pancreas': 'Circulatory system disorders',
                    'Gonarthrosis [arthrosis of knee](FG)': 'Circulatory system disorders',
                    '#Arthrosis': 'Circulatory system disorders',
                    'Dorsalgia': 'Summary diagnoses',
                    '#Other joint disorders': 'Summary diagnoses',
                    'Diseases of the nervous system': 'Nervous system disorders',
                    'Diseases of the musculoskeletal system and connective tissue': 'Circularoty system disorders',
                    'Injury, poisoning and certain other consequences of external causes': 'Summary diagnoses',
                    'Diseases of the digestive system': 'Summary diagnoses',
                    'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified': 'Summary diagnoses',
                    }
    # adding the general group of each PRS
    high_sig['phen_group'] = high_sig['Phenotype Description'].apply(lambda x: prefix_group.get(x.split(':')[0], None))
    # list of our interesting PRS
    my_phengroup = ['Family history', 'Body composition', 'Medication', 'Blood count', 'Medical conditions',
                    'Autorefraction',
                    'Blood biochemistry', 'Circulatory system disorders', 'Intraocular pressure', 'Summary Diagnoses',
                    'Sleep',
                    'Spirometry', 'Pain', 'Eyesight', 'Body size measures', 'Hearing',
                    'Mouth', 'Sun exposure', 'Urine assays', 'Blood pressure',
                    'General health', 'Nervous system disorders', 'Sexual factors', 'Reaction time',
                    'Arterial stiffness', 'Chest pain', 'Breathing', 'Claudication and peripheral artery disease',
                    'Prospective memory', 'Medical information']

    # filtering the PRS to only relevant ones which are also segnificant to genetics
    phen_group_bool = []
    for i in high_sig['phen_group']:
        phen_group_bool.append(i in my_phengroup)
    high_sig['phen_group_bool'] = phen_group_bool
    # list of the ralvant and corralted to genetics PRS
    sig = high_sig.index[high_sig['phen_group_bool']].to_list()
    # removing duplicated of PRS's name
    siguniqe = list(dict.fromkeys(sig))
    return (siguniqe)


def write_getsigunique_cache(dir="/net/mraid20/export/jasmine/zach/prs_associations/",
                             fileName="getsigunique_cache.csv", include_all=False):
    pd.Series(getsigunique(cached=False, include_all=include_all)).to_csv(dir + fileName)
