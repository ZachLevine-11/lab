
def compare_rna_exp_genes(counts_df, loaderdata, cbc_df, cbc_df_age, picked = "n_most_var", thresholds =  [0.5], response = "age", normalize = True, sizes_train_prop = [1], n_genes = None):
    data = counts_df.iloc[:, :-1].clip(1e-5)
    data = np.log10(data)
    print(len(counts_df))
    #0.08: 1400, avg mean highest
    ##jacob used 0.5, I originally used 0.8
    i = 0
    total = 2 * len(thresholds) * len(sizes_train_prop)
    cbc_df = cbc_df.drop(["Date"], axis=1).dropna()
    reses = {}
    combined_indices = list(set(cbc_df.index).intersection(set(data.index)))#.intersection(set(loaderdata.index))
    print(len(combined_indices))
    if response != "age":
        Y = cbc_df.loc[combined_indices, :][response]
    else:
        cbc_df_age["real_age"] = cbc_df_age[ "age"]  ##there's already an age column in the loaderdata, so make sure you use the one from the rnaseq
        ##Furthermore, since that column already exists, when you merge it will suffix both, and the "age" column won't exist
        Y = cbc_df_age.loc[combined_indices,:]["real_age"]
    X_train_CBC, X_test_CBC, _, _ = do_split_and_normalize_eff(cbc_df.loc[combined_indices, :], Y, normalize=normalize)
    for thresh in thresholds:
        if picked == 'n_most_expressed':
            picked_genes = list(data.mean().sort_values(ascending=False)[:n_genes].index)
        elif picked == 'n_most_var':
            picked_genes = list(data.std().sort_values(ascending=False)[:n_genes].index)
        elif picked == "most_var":
            picked_genes = list(data.std()[data.std() > data.std().mean() + thresh * data.std().std()].index)
        elif picked == "random":
            picked_genes = np.random.choice(list(filter(lambda x: data.loc[:, x].std()> 1e-8 if data.loc[:, x].dtype == "float64" else False, data.columns)), 700)
        elif picked == "most_expressed":
            picked_genes = list(data.std()[data.mean() > data.mean().mean() + thresh*data.mean().std()].index)
        print("ok")
      #  loaderdata = loaderdata.loc[~loaderdata.index.duplicated(keep="last"), :]
        X_train_RNA, X_test_RNA, Y_train_RNA, Y_test_RNA = do_split_and_normalize_eff(data.loc[combined_indices, picked_genes], Y, normalize = normalize, inds_train = X_train_CBC.index)
        X_train_combined, X_test_combined = X_train_RNA.merge(X_train_CBC, how = "inner", left_index = True, right_index = True), X_test_RNA.merge(X_test_CBC, how = "inner", left_index = True, right_index = True)
        print(len(X_train_combined), len(X_test_combined))
        models = [Lasso, Ridge, XGBRegressor] #XGBRegressor
        model_names = {Lasso: "Lasso", Ridge: "Ridge", XGBRegressor: "XGBoost"}
        for train_size in sizes_train_prop:
            for model in models:
                print(100*i/total, "% finished")
                if model != XGBRegressor:
                    fitted_model_RNA = model(fit_intercept=True, alpha = 0.1).fit(X_train_combined.loc[:, picked_genes], Y_train_RNA)
                    fitted_model_CBC = model(fit_intercept=True, alpha = 0.1).fit(X_train_combined.loc[:, cbc_df.columns], Y_train_RNA)
                    fitted_model_both = model(fit_intercept=True, alpha=0.1).fit(X_train_combined.loc[:, picked_genes + list(cbc_df.columns)],
                                                                                Y_train_RNA)
                else:
                    fitted_model_RNA = model(max_depth=7, n_estimators=1000, learning_rate = 0.05).fit(X_train_combined.loc[:, picked_genes], Y_train_RNA) #eta 0.001 if this doesn't work
                    fitted_model_CBC = model(max_depth=7, n_estimators=1000, learning_rate = 0.05).fit(X_train_combined.loc[:, cbc_df.columns], Y_train_RNA)
                    fitted_model_both = model(max_depth=7, n_estimators=1000, learning_rate = 0.05).fit(X_train_combined.loc[:, picked_genes + list(cbc_df.columns)], Y_train_RNA)

                fitted_model_predictions_RNA = fitted_model_RNA.predict(X_test_combined.loc[:, picked_genes])
                fitted_model_predictions_CBC = fitted_model_CBC.predict(X_test_combined.loc[:, cbc_df.columns])
                fitted_model_predictions_both = fitted_model_both.predict(X_test_combined.loc[:, picked_genes + list(cbc_df.columns)])

                res_RNA = pd.DataFrame({"pred": fitted_model_predictions_RNA.flatten(), "true": Y_test_RNA.to_numpy().flatten()})
                res_CBC = pd.DataFrame({"pred": fitted_model_predictions_CBC.flatten(), "true": Y_test_RNA.to_numpy().flatten()})
                res_both = pd.DataFrame({"pred": fitted_model_predictions_both.flatten(), "true": Y_test_RNA.to_numpy().flatten()})

                reses[(model_names[model], train_size, thresh, len(picked_genes))] = {"RNA_Corr": res_RNA.corr().iloc[0,1], "CBC_Corr" : res_CBC.corr().iloc[0,1],"both_Corr" : res_both.corr().iloc[0,1]}
                i += 1
    reses = pd.DataFrame(reses).T
    reses.index.names = ["model", "train_size", "RNA_variability_thresh", "number_of_picked_genes"]
    return reses

# res_mean = {}
# res_var = {}

#for n_genes in [500,1000,2000,5000]: res_mean[n_genes] = compare_rna_exp_genes(counts_df, loaderdata, cbc_df, cbc_df_age, picked = "n_most_expressed", thresholds =  [0.5], response = "age", normalize = True, sizes_train_prop = [1], n_genes = n_genes)
#for n_genes in [500,1000,2000,5000]: res_var[n_genes] = compare_rna_exp_genes(counts_df, loaderdata, cbc_df, cbc_df_age, picked = "n_most_var", thresholds =  [0.5], response = "age", normalize = True, sizes_train_prop = [1], n_genes = n_genes)

# res
def plot_reses(res)
make_plot = True
    if make_plot:
        #reses_tde1 = compare_rna_exp_genes(counts_no_replicates_tde1, loaderdata, bt_df_cbc, bt_df_cbc_age)
        #reses_nextera = compare_cbc_and_rna_exp(counts_no_replicates_nextera, loaderdata, bt_df_cbc, bt_df_cbc_age)
        reses_tde1 = res
        reses_tde1["Method"] = "TDE1"
        #reses_nextera["Method"] = "nextera"
        reses_both = reses_tde1
        reses_both["Type"] = reses_both.apply(
            lambda row: str(row["number_of_picked_genes"]) + " " + str(
                row["variable"]), axis=1)
        for model in reses_both["model"].unique():
            for method in reses_both["Method"]:
                reses_both_eff = reses_both.loc[reses_both["model"] == model,:]
                reses_both_eff = reses_both_eff.loc[reses_both_eff["Method"] == method,:]
                colours = [cm.jet(x) for x in np.linspace(0, 1, len(reses_both_eff["Type"].unique()))]
                plt.figure(figsize = (15,15))
                for i in range(len(reses_both_eff["Type"].unique())):
                    kind = reses_both_eff["Type"].unique()[i]
                    df = reses_both_eff.loc[reses_both_eff["Type"] == kind,:]
                    plt.plot(df["train_size"], df["value"], c= colours[i], label = kind)
                plt.legend()
                plt.title(model + "" + str(method))
                plt.xlabel("Gene Count (n Most Variable)")
                plt.ylabel("Prediction Correlation to True Values")
                #os.chdir("/home/zacharyl/")
                plt.savefig(str(model) + "" + str(method) + ".jpg")
                plt.close()

def mean_exp_plot:
    for n_genes in [500,1000,2000,5000]: res[n_genes] = compare_rna_exp_genes(counts_df, loaderdata, cbc_df, cbc_df_age, picked = "n_most_expressed", thresholds =  [0.5], response = "age", normalize = True, sizes_train_prop = [1], n_genes = n_genes)
    reses_mean = pd.concat([res[500], res[1000],res[2000],res[5000]])
    reses = reses_mean.reset_index(inplace = False)
    fig = px.line(reses, x='number_of_picked_genes', y='RNA_Corr', color='model',title = "RNA Correlation as a function of n most expressed genes",)
    fig.write_image("mean_genes.jpg")
def var_expr_plot:
    for n_genes in [500,1000,2000,5000]: res[n_genes] = compare_rna_exp_genes(counts_df, loaderdata, cbc_df, cbc_df_age, picked = "n_most_var", thresholds =  [0.5], response = "age", normalize = True, sizes_train_prop = [1], n_genes = n_genes)
    reses_var = pd.concat([res[500], res[1000],res[2000],res[5000]])
    reses = reses_var.reset_index(inplace = False)
    fig = px.line(reses, x='number_of_picked_genes', y='RNA_Corr', color='model',title = "RNA Correlation as a function of n most variably expressed genes",)
    fig.write_image("var_genes.jpg")

# for n_genes in [500,1000,2000,5000]: res[n_genes] = compare_rna_exp_genes(counts_df, loaderdata, cbc_df, cbc_df_age, picked = "n_most_var", thresholds =  [0.5], response = "age", normalize = True, sizes_train_prop = [1], n_genes = n_genes)
# reses_var = pd.concat([res[500], res[1000],res[2000],res[5000]])
# reses = reses_var.reset_index(inplace = False)
# )
# fig = px.line(reses, x='number_of_picked_genes', y=('both_Corr', "RNA_Corr", "CBC_Corr"), color='model',title = "RNA Correlation as a function of n most variably expressed genes",)
if __name__ == "__main__":
    res = {}
    for n_genes in [500,750, 1000, 1500, 2000,2500, 3000, 4000, 4500, 5000, 5500]:
        res[n_genes] = compare_rna_exp_genes(counts_no_replicates_tde1, loaderdata, cbc_df, cbc_df_age, picked = picked_genes,
                                         thresholds =  [0.5], response = "age", normalize = True, sizes_train_prop = [1], n_genes = n_genes)
    reses_var = pd.concat([[res[500],res[750], res[1000], res[1500], res[2000],res[2500], res[3000], res[4000], res[4500], res[5000], res[5500]]).reset_index()

    #reses_var = pd.concat([res[500], res[1000]]).reset_index()
    reses_melted = reses_var.melt(["model", "number_of_picked_genes", "RNA_variability_thresh", "train_size"])
    for val in reses_melted.model.unique():
        temporary = reses_melted.loc[reses_melted.model == val,:]

        fig = px.line( temporary, x='number_of_picked_genes', y="value", color='variable',
                   title="RNA Correlation as a function of n most variably expressed genes", )
        fig.write_image(val+".jpg")
