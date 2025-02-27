##### from nastya
bt_df = BloodTestsLoader().get_data().df
bt_df_meta = BloodTestsLoader().get_data().df_metadata
cbc_lab_ind = bt_df_meta[bt_df_meta['kupat_holim'] == 'tenk-cbc'].index
bt_df_cbc = bt_df.loc[cbc_lab_ind].dropna(how='all', axis=1)
bt_df_cbc_age = bt_df_meta.loc[cbc_lab_ind]['age'].reset_index().set_index("RegistrationCode")
bt_df_cbc = bt_df_cbc.reset_index(drop=False).set_index("RegistrationCode")
bt_df_cbc_age = bt_df_cbc_age.loc[~bt_df_cbc_age.index.duplicated(keep = "last"), :]
bt_df_cbc = bt_df_cbc.loc[~bt_df_cbc.index.duplicated(keep = "last"), :]

runs_summary = pd.read_csv("Novaseq_Runs_Summary.csv")
runs_summary["Plates_Included_Numeric"] = list(map(lambda x: rack_extractor(x) if x is not None else [0],runs_summary["Plate Info"]))
runs_summary = runs_summary.dropna()

#original_counts = pd.read_csv(base_dir + "corrected_counts.csv") ##corrects for UMI, PCR amplification is not uniform
counts_no_replicates, total_counts_no_reps = correct_counts(original_counts, metadata_df, runs_summary, True)
counts_no_replicates_tde1 = counts_no_replicates.loc[list(map(lambda x:  str(x) == "TDE1 (1uL TDE1, 14 cycles amplification)" , counts_no_replicates["Methodology"])),:]
counts_no_replicates_nextera = counts_no_replicates.loc[list(map(lambda x:  str(x) == "Nextera XT" , counts_no_replicates["Methodology"])),:]

counts_with_replicates, total_counts_with_reps = correct_counts(original_counts, metadata_df, runs_summary, True)

def compare_cbc_and_rna_exp(counts_df, loaderdata, cbc_df, cbc_df_age, picked = "most_var", response = "age"):
    not_missing_cbc_cols = list(cbc_df.isna().sum().loc[cbc_df.isna().sum() <= len(cbc_df)*0.5].index)
    not_missing_cbc_cols.remove("Date") ##this column doesn't have missing values, but don't fit on date
    data = counts_df.iloc[:, :-1].clip(1e-5)
    data = np.log10(data)
    #0.08: 1400, avg mean highest
    if picked == "most_var":
        picked_genes = list(data.std()[data.std() > data.std().mean() + 0.08*data.std().std()].index)
    elif picked == "random":
        picked_genes = np.random.choice(list(filter(lambda x: data.loc[:, x].std()> 1e-8 if data.loc[:, x].dtype == "float64" else False, data.columns)), 700)
    elif picked == "most_expressed":
        picked_genes = list(data.std()[data.mean() > data.mean().mean() + 0.5*data.mean().std()].index)
    rna_df = data.loc[:, picked_genes]
    cbc_df = cbc_df.loc[:, not_missing_cbc_cols].dropna()
    loaderdata = loaderdata.loc[~loaderdata.index.duplicated(keep="last"), :]
    data_all = cbc_df.merge(rna_df.merge(loaderdata, right_index=True, left_index=True, how='inner'), right_index=True, left_index=True, how='inner')
    if response != "age":
        Y = data_all[response]
    else:
        cbc_df_age["real_age"] = cbc_df_age["age"] ##there's already an age column in the loaderdata, so make sure you use the one from the rnaseq
        ##Furthermore, since that column already exists, when you merge it will suffix both, and the "age" column won't exist
        Y = cbc_df_age.merge(data_all, left_index = True, right_index = True, how = "inner")["real_age"]
    common_ids = list(set(data_all.index).intersection(set(cbc_df_age.index)))
    X_train, X_test, Y_train, Y_test, scaler_X, scaler_Y = do_split_and_normalize(data_all.loc[common_ids, picked_genes + not_missing_cbc_cols], Y.loc[common_ids])
    reses = {}
    for model in [Lasso, Ridge, XGBRegressor]:
        if model in [Lasso,Ridge]:
            fitted_model_RNA = model(fit_intercept=True, alpha = 0.1).fit(X_train.loc[:, picked_genes], Y_train)
            fitted_model_CBC = model(fit_intercept=True, alpha = 0.1).fit(X_train.loc[:, not_missing_cbc_cols], Y_train)
            fitted_model_both = model(fit_intercept=True, alpha = 0.1).fit(X_train.loc[:, picked_genes + not_missing_cbc_cols], Y_train)

            fitted_model_predictions_RNA = fitted_model_RNA.predict(X_test.loc[:, picked_genes])
            fitted_model_predictions_CBC = fitted_model_CBC.predict(X_test.loc[:, not_missing_cbc_cols])
            fitted_model_predictions_both = fitted_model_both.predict(X_test.loc[:, picked_genes + not_missing_cbc_cols])

            res_RNA = pd.DataFrame({"pred": fitted_model_predictions_RNA.flatten(), "true": Y_test.to_numpy().flatten()})
            res_CBC = pd.DataFrame({"pred": fitted_model_predictions_CBC.flatten(), "true": Y_test.to_numpy().flatten()})
            res_both = pd.DataFrame({"pred": fitted_model_predictions_both.flatten(), "true": Y_test.to_numpy().flatten()})

            reses[model] = {"RNA": res_RNA, "CBC" : res_CBC, "both" : res_both}
        elif model in [XGBRegressor]:
            # read data
            bst_model_gx = model(max_depth=4, num_estimators=400, bagging_fraction=0.7, bagging_freq=1, feature_fraction=.2, eta =.001,metric='auc')
            bst_model_cbc = model(max_depth=4, num_estimators=400, bagging_fraction=0.7, bagging_freq=1, feature_fraction=.2, eta =.001,metric='auc')
            bst_model_both = model(max_depth=4, num_estimators=400, bagging_fraction=0.7, bagging_freq=1, feature_fraction=.2, eta =.001,metric='auc')

            bst_model_cbc = bst_model_cbc.fit(X_train.loc[:, not_missing_cbc_cols], Y_train)
            bst_model_gx = bst_model_gx.fit(X_train.loc[:, picked_genes], Y_train)
            bst_model_both = bst_model_both.fit(X_train.loc[:, picked_genes + not_missing_cbc_cols], Y_train)

            bst_model_predictions_CBC = bst_model_cbc.predict(X_test.loc[:, not_missing_cbc_cols])
            bst_model_predictions_RNA = bst_model_gx.predict(X_test.loc[:, picked_genes])
            bst_model_predictions_both = bst_model_both.predict(X_test.loc[:, picked_genes + not_missing_cbc_cols])

            res_RNA = pd.DataFrame({"pred": bst_model_predictions_RNA.flatten(), "true": Y_test[0]})
            res_CBC = pd.DataFrame({"pred": bst_model_predictions_CBC.flatten(), "true": Y_test[0]})
            res_both = pd.DataFrame({"pred": bst_model_predictions_both.flatten(), "true": Y_test[0]})

            reses[model] = {"RNA": res_RNA, "CBC" : res_CBC, "both" : res_both}
    return reses


for model in [Lasso, Ridge, XGBRegressor]:
    if model in [Lasso, Ridge]:
        fitted_model_RNA = model(fit_intercept=True, alpha=0.1).fit(X_train.loc[:, picked_genes], Y_train)
        fitted_model_CBC = model(fit_intercept=True, alpha=0.1).fit(X_train.loc[:, not_missing_cbc_cols], Y_train)
        fitted_model_both = model(fit_intercept=True, alpha=0.1).fit(
            X_train.loc[:, picked_genes + not_missing_cbc_cols], Y_train)

        fitted_model_predictions_RNA = fitted_model_RNA.predict(X_test.loc[:, picked_genes])
        fitted_model_predictions_CBC = fitted_model_CBC.predict(X_test.loc[:, not_missing_cbc_cols])
        fitted_model_predictions_both = fitted_model_both.predict(X_test.loc[:, picked_genes + not_missing_cbc_cols])

        res_RNA = pd.DataFrame({"pred": fitted_model_predictions_RNA.flatten(), "true": Y_test.to_numpy().flatten()})
        res_CBC = pd.DataFrame({"pred": fitted_model_predictions_CBC.flatten(), "true": Y_test.to_numpy().flatten()})
        res_both = pd.DataFrame({"pred": fitted_model_predictions_both.flatten(), "true": Y_test.to_numpy().flatten()})

        reses[model] = {"RNA": res_RNA, "CBC": res_CBC, "both": res_both}
    elif model in [XGBRegressor]:
        # read data
        bst_model_gx = model(max_depth=4, num_estimators=400, bagging_fraction=0.7, bagging_freq=1, feature_fraction=.2,
                             eta=.001, metric='auc')
        bst_model_cbc = model(max_depth=4, num_estimators=400, bagging_fraction=0.7, bagging_freq=1,
                              feature_fraction=.2, eta=.001, metric='auc')
        bst_model_both = model(max_depth=4, num_estimators=400, bagging_fraction=0.7, bagging_freq=1,
                               feature_fraction=.2, eta=.001, metric='auc')

        bst_model_cbc = bst_model_cbc.fit(X_train.loc[:, not_missing_cbc_cols], Y_train)
        bst_model_gx = bst_model_gx.fit(X_train.loc[:, picked_genes], Y_train)
        bst_model_both = bst_model_both.fit(X_train.loc[:, picked_genes + not_missing_cbc_cols], Y_train)

        bst_model_predictions_CBC = bst_model_cbc.predict(X_test.loc[:, not_missing_cbc_cols])
        bst_model_predictions_RNA = bst_model_gx.predict(X_test.loc[:, picked_genes])
        bst_model_predictions_both = bst_model_both.predict(X_test.loc[:, picked_genes + not_missing_cbc_cols])

        res_RNA = pd.DataFrame({"pred": bst_model_predictions_RNA.flatten(), "true": Y_test[0]})
        res_CBC = pd.DataFrame({"pred": bst_model_predictions_CBC.flatten(), "true": Y_test[0]})
        res_both = pd.DataFrame({"pred": bst_model_predictions_both.flatten(), "true": Y_test[0]})

        reses[model] = {"RNA": res_RNA, "CBC": res_CBC, "both": res_both}