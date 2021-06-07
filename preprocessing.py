import numpy as np
import pandas as pd
import numpy as np
from sklearn import linear_model
from statistics import mean
from neuroCombat import neuroCombat

def connectivity_test(neuro_data, clinical_data):
    """
    Apply a connectivity test to make sure that all the conections among controls are relevant 
    (relevant means that more than 60% of the controls have connections different than 0)

    Arguments:
    ---------

    neuro_data: neuroimaging data

    clinical_data: The only arguments that it must contain is controls_ms

    Returns:
    --------

    A list with all the variables that succesfully passed the test
    """
    # Get controls id
    ids = [name for name in clinical_data[clinical_data.controls_ms == 0].index]
    controls = neuro_data.loc[ids]

    controls[controls > 0.1] = 1
    mask = controls.sum(axis = 0, skipna = True)
    #bellow 60% values of HC need to be removed the connection
    thr = round(controls.shape[0]*0.6)
    mask[mask < thr] = 0
    mask[mask >= thr] = 1

    passed_control = list()

    for i,m in enumerate(mask):
        if m == 1:
            passed_control.append(i)

    return passed_control

def linear_correction(clinical_data, neuro_data):
    """
    Adjust the matrix data per age and sex.

    Arguments:
    ---------

    clinical_data: The only two arguments that it must contain is age and sex

    neuro_data: neuroimaging data

    Returns:
    --------

    Pandas dataframe with the neuroimaging information corrected and the clinical data
    """
    # create an empty array
    corrected_data = np.zeros(neuro_data.shape)

    X = np.array(list(zip(clinical_data['age'], clinical_data['sex'])))
    #Correct for age and sex the connectivity matrices
    for i in range(neuro_data.shape[1]):
        Y = neuro_data.values[:,i] #Extract the values inside matrix
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)
        y_pred = regr.predict(X)
        residual = (Y - y_pred)
        val_corrected = residual + mean(Y)
        # save values
        corrected_data[:,i] = val_corrected
    
    corrected_data[corrected_data < 0.1] = 0

    return pd.concat([pd.DataFrame(corrected_data, index = neuro_data.index, columns = neuro_data.columns), clinical_data], join="inner", axis=1)

def harmonization(neuro_data, clinical_data):
    """
    Takes raw data and harmonizes it by using the Combat method.

    Arguments:
    ---------

    neuro_data: pandas dataframe or numpy array with the information related to neuroimaging or ROIs. The shape of the matrix has to be (Samples, Features)

    clinical_data: It contains data that we want to take into account in the harmonization process, rather if it's because we don't want to affect it 
    influence (like age or sex) or because we want to modify it's influence, scanner effects.

    batch_col: is the name of the column in "clinical_data" that we want to modify it's influence in the data.

    Returns:
    --------

    Pandas dataframe with the harmonized neuroimaging data concatenated with the clinical data provided
    
    """
    dat = neuro_data.transpose()
    
    index = [name for name in neuro_data.index]
    scanner = [0 if "MSVIS" in name else 1 for name in index]

    clinical_data["scanner"] = scanner

    har_data = neuroCombat.neuroCombat(dat = dat,
                                       covars = clinical_data,
                                       batch_col = "scanner")

    clinical_data.drop(["scanner"], axis=1, inplace=True)

    return pd.concat([pd.DataFrame(har_data["data"].transpose(), index = neuro_data.index, columns = neuro_data.columns), clinical_data], 
                     join = "inner", 
                     axis=1)

def data_augmentation(X, y):

    sm = SMOTE(random_state=42)
    X_res, Y_res = sm.fit_resample(X, y)

    df_smote_over = pd.concat([pd.DataFrame(X_res), pd.DataFrame(Y_res)], axis=1, join="inner")

    return df_smote_over

def wm_pipeline(neuro_data, clinical_data, index, tolerance, name, target=-1, mw=True):

    wm_conec = neuro_data.iloc[:, index]

    # age and sex corrections
    wm_correct = linear_correction(clinical_data, wm_conec)
    wm_imputed = outlier_imputation(wm_correct.iloc[:, :-5])

    # harmonization
    wm_har = harmonization(wm_imputed, clinical_data)

    # Statistical Significance

    if mw:
        # wm_final = statistically_significant_variables(wm_har, tolerance, name)
        wm_final = types_diff(wm_har)
        wm_final = data_augmentation(wm_final.iloc[:, :-8], wm_final.iloc[:, -1])

        return wm_final.iloc[:, :-1], wm_final.iloc[:, -1]
    
    if target != -1:
        wm_har = wm_har[wm_har.controls_ms == 1]

    X = wm_har.iloc[:, :-5]
    y = wm_har.iloc[:, target]

    rf = RandomForestRegressor(n_estimators=1000, max_depth=2)
    rf.fit(X, y)

    f_imp = pd.Series(rf.feature_importances_)
    importance_sorted = pd.DataFrame(list(f_imp), index=list(X.columns),columns=["Feature Importance"])
    importance_sorted = importance_sorted.sort_values('Feature Importance', ascending=False)[:100]

    cols = [col for col in importance_sorted.index]

    return X.loc[:, cols], y

def gm_pipeline(neuro_data, clinical_data, index, tolerance, name, target=-1, mw=True):

    gm_conec = neuro_data.iloc[:, index]

    # age and sex corrections
    gm_correct = linear_correction(clinical_data, gm_conec)
    gm_imputed = outlier_imputation(gm_correct.iloc[:, :-5])
    gm_imputed = pd.concat([gm_imputed, gm.iloc[:, -5:]], axis=1, join="inner")

    # Statistical Significance

    if mw:
        # gm_final = statistically_significant_variables(gm_imputed, tolerance, name)
        gm_final = types_diff(gm_imputed)
        gm_final = data_augmentation(gm_final.iloc[:, :-8], gm_final.iloc[:, -1])

        return gm_final.iloc[:, :-1], gm_final.iloc[:, -1]

    if target != -1:
        gm_imputed = gm_imputed[gm_imputed.controls_ms == 1]

    X = gm_imputed.iloc[:, :-5]
    y = gm_imputed.iloc[:, target]

    rf = RandomForestRegressor(n_estimators=1000, max_depth=2)
    rf.fit(X, y)

    f_imp = pd.Series(rf.feature_importances_)
    importance_sorted = pd.DataFrame(list(f_imp), index=list(X.columns),columns=["Feature Importance"])
    importance_sorted = importance_sorted.sort_values('Feature Importance', ascending=False)[:100]

    cols = [col for col in importance_sorted.index]

    return X.loc[:, cols], y

def classifier_pipeline(fa, gm, func):

    index = connectivity_test(fa.iloc[:, :-5], fa.iloc[:, -5:])

    X_fa, y = wm_pipeline(fa.iloc[:, :-5], fa.iloc[:, -5:], index, 0.05, "FA Data")
    X_gm, y = gm_pipeline(gm.iloc[:, :-5], gm.iloc[:, -5:], index, 0.05, "GM Data")
    X_func, y = gm_pipeline(func.iloc[:, :-5], func.iloc[:, -5:], index, 0.05, "FUNC Data")

    X = pd.concat([X_fa, X_gm, X_func], join="inner", axis=1)

    return X, y

def regression_pipeline(fa, gm, func, target):

    index = connectivity_test(fa.iloc[:, :-5], fa.iloc[:, -5:])

    X_fa, y = wm_pipeline(fa.iloc[:, :-5], fa.iloc[:, -5:], index, 0.05, "FA Data", target = target, mw = False)
    X_gm, y = gm_pipeline(gm.iloc[:, :-5], gm.iloc[:, -5:], index, 0.05, "GM Data", target = target, mw = False)
    X_func, y = gm_pipeline(func.iloc[:, :-5], func.iloc[:, -5:], index, 0.05, "FUNC Data", target = target, mw = False)

    X = pd.concat([X_fa, X_gm, X_func], join="inner", axis=1)

    return X, y

# Kamila

def stats_data(data):
    patients_fa = data.loc[data["controls_ms"] == 1].copy()
    patients_fa.drop(labels = ["controls_ms", "age", "sex", "dd", "edss"], axis = 1, inplace = True)
    controls_fa = data.loc[data["controls_ms"] == 0].copy()
    controls_fa.drop(labels =  ["controls_ms", "age", "sex", "dd", "edss"], axis = 1, inplace = True)
    feats = data.iloc[:, :-5].columns.to_list()

    return patients_fa, controls_fa, feats

def MW_U(data):
    patients_fa, controls_fa, feats = stats_data(data)
    MannWhitney_tests = pd.DataFrame(columns=['ROI', 'U', 'pvalue'])
    for attr in feats:
        stat,p = mannwhitneyu(patients_fa[attr], controls_fa[attr])
        MannWhitney_tests = MannWhitney_tests.append({'ROI': attr, 'U': stat, 'pvalue': p}, ignore_index=True)

    test, p_corr = fdrcorrection(MannWhitney_tests["pvalue"], alpha = 0.05, method = "indep", is_sorted = False)
    MannWhitney_tests ["Rejected"] = test
    MannWhitney_tests ["p_corr"] = p_corr

    return MannWhitney_tests

def statistically_significant_variables(data, tolerance, name):
    MW = MW_U(data)
    difference_fa = MW[MW["p_corr"] < tolerance]
    if len(difference_fa) <= 5:
        print("No Significant Variables in", name)
        return 0

    different_fa = difference_fa["ROI"].tolist()
    fa_har_clinical = data.loc[:,["age", "sex", "dd", "edss", "controls_ms"]]
    fa_har_t = data[different_fa]
    fa_har_new = pd.merge(fa_har_t, fa_har_clinical, left_index=True, right_index=True)

    return fa_har_new

# Eloy 

def types_diff(data):
    conn_stat = pd.DataFrame(columns=['ROI','pvalue'])
    feats = data.iloc[:, :-1].columns.to_list()
    for connections in feats:
        stat,p = shapiro(data[connections])
        alpha=0.05
        sample1 = data.loc[data["controls_ms"]==0,connections]
        sample2 = data.loc[data["controls_ms"]==1,connections]
        if p > alpha:
            stat, p = bartlett(sample1,sample2)
            homovar = True
            if (p<=0.05):
                homovar = False
            stat,p = ttest_ind(sample1,sample2,equal_var=homovar)
        else:
            stat,p = mannwhitneyu(data.loc[data["controls_ms"]==0,connections],data.loc[data["controls_ms"]==1,connections],alternative='two-sided')
    
        if (p<=0.05):
            conn_stat = conn_stat.append({'ROI': connections,'pvalue': p}, ignore_index=True)
        
    print('Statistically diferences in %d of connections' % len(conn_stat))

    diff = conn_stat.copy()

    p_corr = multipletests(diff["pvalue"], alpha = 0.05, method = "bonferroni", is_sorted = False)
    diff["p_corr"] = p_corr[1] #Added the bonferroni correction
    diff_fdr = diff[diff["p_corr"] < 0.05] #FDR correction with lowest pvalue
    print('Statistically diferences in %d of connections with FDR' % len(diff_fdr))
    diff_fa = diff_fdr["ROI"].tolist()
    fa_clinic = data.loc[:,["age", "sex", "dd", "edss", "controls_ms"]]
    #Adding index fa_har
    # fa_har = data.set_index(data.index) perque???
    fa_har_bonferroni = data[diff_fa]
    fa_har_corr = pd.merge(fa_har_bonferroni, fa_clinic, left_index=True, right_index=True)
    
    return fa_har_corr

def outlier_imputation(neuro_data):
    """
    Deals with the problem of zero-values in data by using linear regression models for imputation.

    Arguments:
    ----------

    neuro_data: data that have to be corrected. Important, do not include Clinical Data.

    Returns:
    --------

    pandas dataframe with the corrected data.
    """
    columns = [col for col in neuro_data.columns]
    df = neuro_data.copy()
    
    model = LinearRegression()

    for col in columns:
        if df[df[col] == 0].shape[0] != 0:
            X_train, y_train = df[df[col] != 0].drop([col], axis = 1), df[df[col] != 0][col]
            X_test = df[df[col] == 0].drop([col], axis = 1)

            model = LinearRegression()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            df.loc[X_test.index, col] = pred
    return df