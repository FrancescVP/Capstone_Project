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