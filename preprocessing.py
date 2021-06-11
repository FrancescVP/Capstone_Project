import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statistics import mean
from neuroCombat import neuroCombat
from imblearn.over_sampling import SMOTE
from scipy.stats import mannwhitneyu, ttest_ind, bartlett, shapiro
from statsmodels.stats.multitest import fdrcorrection, multipletests


def classifier_pipeline(fa, gm, func):
    """
    This function is in charge of executing the entire data post-processing process.
    All functions in this scripts are ran in this pipeline, from running the
    connectivity test to data augmentation.
    Arguments:
    ---------
    fa: Fractional Anisotropy Data, containing FA-weighted connectivity matrix with information related to White Matter (WM).
    gm: Structural Gray Matter Brain Network Data, GM connectivity network 
    func: Resting State fMRI Data, containing data from brain signal correlation/synchronization through resting-state functional connectivity
    Returns:
    --------
    X: Pandas data frame containing the most statistically significant variables in order to describe the behavior of the target. 
    This data frame has been corrected from biases such as sex, age, and "scanner effects". A data augmentation process has also 
    been applied to deal with the imbalance of the target, which contains few Health Subjects.
    y: Target (controls_ms)
    """

    index = connectivity_test(fa.iloc[:, :-5], fa.iloc[:, -5:])

    X_fa, y = wm_pipeline(fa.iloc[:, :-5], fa.iloc[:, -5:], index)
    X_gm, y = gm_pipeline(gm.iloc[:, :-5], gm.iloc[:, -5:], index)
    X_func, y = gm_pipeline(func.iloc[:, :-5], func.iloc[:, -5:], index)

    X = pd.concat([X_fa, X_gm, X_func], join="inner", axis=1)

    df_final = data_augmentation(X, y)

    return df_final.iloc[:, :-1], df_final.iloc[:, -1]


def wm_pipeline(neuro_data, clinical_data, index):
    """
    All the post-processing data process applied to White Matter matrices are contained
    in this pipeline (age and gender correction, harmonization, feature selection, and
    data augmentation)
    Arguments:
    ---------
    neuro_data: Fractional Anisotropy Data, containing FA-weighted connectivity matrix with information related to White Matter (WM).
    clinical_data: self-explanatory.
    index: list containing all the names from the features that successfully passed the connectivity test.
    Returns:
    --------
    wm_final.iloc[:, :-1]: Pandas dataframe with the post-processed features
    
    wm_final.iloc[:, -1]: Pandas dataframe with the target variable.
    """

    wm_conec = neuro_data.iloc[:, index]

    # age and sex corrections
    wm_correct = linear_correction(clinical_data, wm_conec)
    wm_imputed = outlier_imputation(wm_correct.iloc[:, :-5])

    # harmonization
    wm_har = harmonization(wm_imputed, clinical_data)

    # Statistical Significance

    # wm_final = statistically_significant_variables(wm_har, tolerance, name)
    wm_final = types_diff(wm_har)

    return wm_final.iloc[:, :-8], wm_final.iloc[:, -1]


def gm_pipeline(neuro_data, clinical_data, index):
    """
    All the post-processing data process applied to Gray Matter matrices are contained
    in this pipeline (age and gender correction, feature selection, and data augmentation)
    Arguments:
    ---------
    neuro_data: Structural Gray Matter Brain Network Data or Resting State fMRI Data.
    clinical_data: self-explanatory.
    index: list containing all the names from the features that successfully passed the connectivity test.
    Returns:
    --------
    gm_final.iloc[:, :-1]: Pandas dataframe with the post-processed features
    
    gm_final.iloc[:, -1]: Pandas dataframe with the target variable.
    """

    gm_conec = neuro_data.iloc[:, index]

    # age and sex corrections
    gm_correct = linear_correction(clinical_data, gm_conec)
    gm_imputed = outlier_imputation(gm_correct.iloc[:, :-5])
    gm_imputed = pd.concat([gm_imputed, clinical_data], axis=1, join="inner")

    # Statistical Significance

    # gm_final = statistically_significant_variables(gm_imputed, tolerance, name)
    gm_final = types_diff(gm_imputed)

    return gm_final.iloc[:, :-8], gm_final.iloc[:, -1]

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
        Y = np.array(neuro_data.values[:,i],dtype = float) #Extract the values inside matrix
        regr = LinearRegression()
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


def stats_data(data):
    """
    Split the data into two different datasets, one for patients and the other one for healthy subjects.
    Arguments:
    ---------
    data: neuro_data + clinical_data
    Returns:
    --------
    patients_fa: data related to patients
    controls_fa: data related to controls
    feats: clinical_data
    """
    patients_fa = data.loc[data["controls_ms"] == 1].copy()
    patients_fa.drop(labels = ["controls_ms", "age", "sex", "dd", "edss"], axis = 1, inplace = True)
    controls_fa = data.loc[data["controls_ms"] == 0].copy()
    controls_fa.drop(labels =  ["controls_ms", "age", "sex", "dd", "edss"], axis = 1, inplace = True)
    feats = data.iloc[:, :-5].columns.to_list()

    return patients_fa, controls_fa, feats


def MW_U(data):
    """
    Mann Whitney test corrected with fdrcorrection
    Arguments:
    ----------
    data: neuro_data + clinical_data
    Returns:
    -------
    pandas dataframe with the information related to the Mann Whitney test.
    """
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
    """
    Pipeline containing the Mann Whitney test.
    Arguments:
    ----------
    data: neuro_data + clinical_data
    tolerance: number between [1-0] for the p-value restriction (example 0.05)
    name: Name of the data matrix, just for printing purposes.
    Returns:
    --------
    pandas dataframe containind the statistically significant variables as a result from the Mann Whitney test.
    """
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


def types_diff(data):
    """
    Statistically significance correction by using bonferroni.
    Arguments:
    ---------
    data: neuro_data + clinical_data
    Returns:
    -------
    pandas dataframe with the features that passed the bonferroni test
    """
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


def data_augmentation(X, y):
    """
    performing data augmentation by using the SMOTE library, which created new instances by using KNN algorithm.
    Arguments:
    ---------
    X: Independent variables
    y: Target
    Returns:
    -------
    X: data-augmented independent variables
    y: data-augmented target
    """

    sm = SMOTE(random_state=42)
    X_res, Y_res = sm.fit_resample(X, y)

    df_smote_over = pd.concat([pd.DataFrame(X_res), pd.DataFrame(Y_res)], axis=1, join="inner")

    return df_smote_over