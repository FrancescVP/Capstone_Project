import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

def harmonization(neuro_data, clinical_data, batch_col):
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
    
    index = [name for name in fa.index]
    scanner = [0 if "MSVIS" in name else 1 for name in index]

    clinical_data["scanner"] = scanner

    har_data = neuroCombat.neuroCombat(dat = dat,
                                       covars = clinical_data,
                                       batch_col = "scanner")

    clinical_data.drop(["scanner"], axis=1, inplace=True)

    return pd.concat([pd.DataFrame(har_data["data"].transpose(), index = neuro_data.index, columns = neuro_data.columns), clinical_data], 
                     join = "inner", 
                     axis=1)