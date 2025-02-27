import pandas as pd
import os
import statsmodels.api as sm
import numpy as np
import lightgbm as lgb

def pred(df, ys, train_indices, test_indices):
    df = df.loc[:, ~df.T.duplicated()]
    X = df.loc[:, list(filter(lambda x: x not in ys, df.columns))].dropna()
    X = sm.add_constant(X)
    X_train = X.iloc[train_indices, :]
    X_test = X.iloc[test_indices, :]
    lens = {}
    res = {}
    for y in ys:
        model = lgb.LGBMRegressor().fit(y = df.iloc[train_indices,:][y], X = X_train)
        res[y] = np.corrcoef(model.predict(X_test), df.iloc[test_indices,:][y].values)[0,1]**2
        print(res[y])
        lens[y] =  len( df.iloc[test_indices,:][y].values)
    return res, lens

if __name__ == "__main__":
    dir = "/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/rna_options/"
    files = list(filter(lambda x: x.endswith(".df"), os.listdir(dir)))
    is_new_dict = {'TDE1 (1uL TDE1, 14 cycles amplification)': 1, 'Nextera XT': 0,
                   'Nextera XT+ TDE1_Exp3 (3uL TDE1, 13 cycles amplification)': 1,
                   'Nextera XT+ TDE1_Exp1 (3.6uL TDE1, 14 cycles amplification)': 0,
                   'Nextera XT+ TDE1_Exp2 (3.6uL TDE1, 14 cycles amplification)': 0}
    metadata = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jasmine/RNA/" + "metadata_with_techn.csv").set_index("index")
    metadata['is_new'] = metadata['methodology'].map(is_new_dict).fillna(0)
    metadata['new'] = metadata['methodology'].map(is_new_dict).astype(str)
    metadata['RegistrationCode'] = metadata['participant_id'].apply(lambda x: '10K_' + str(int(x)))
    metadata = metadata.set_index("RegistrationCode")
    baseline_values= pd.read_csv('/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/for_review/baseline_values.csv', index_col=0)
    clin=baseline_values[['age', 'bmi', 'bt__lymphocytes_abs', 'bt__lymphocytes_abs','bt__neutrophils_abs']]
    dfs = {}
    reses = {}
    lens = {}
    for file in files:
        dfs[file] = pd.read_pickle(dir + file)
        dfs[file] = dfs[file][dfs[file].index.isin(metadata.index)].merge(clin, left_index = True, right_index = True).dropna()
        train_indices = np.random.choice(a = list(range(len(dfs[file].index))) , size = int(np.floor(len(dfs[file].index)*0.1)), replace = False)
        test_indices = list(set(list(range(len(dfs[file].index)))) - set(train_indices))
        reses[file], lens[file] = pred(dfs[file], list(clin.columns.values), train_indices, test_indices)
    reses = pd.DataFrame(reses)

