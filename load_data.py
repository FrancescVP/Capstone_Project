from zipfile import ZipFile
import pandas as pd
import numpy as np
import os
import re

def unzip_data(path = "data", file = "data.zip"):
    """
    Since the data is divided in to many folders to be stored in a github repository,
    this function exctract all the documents from a .zip file, which is easier to store.

    Arguments
    ---------

    path : the path in which the user prefers to store the data.

    file : the .zip file from which we'll extract the data

    Returns
    -------

    Nothing
    """
    os.mkdir(path)

    with ZipFile(file, 'r') as zip:
        zip.extractall("data")
        print("Done!")

def data_loader(path = "data/", unzip = True):

    """
    data_loader is the main function of this script, it calls all the other functions. The main task consist in
    grouping the data compressed in the zip file into useful dataframes. It also performs a connectivity test 
    and data harmonization if needed.

    Arguments
    ---------

    path : the path in which the data is stored.

    unzip : a boolean variable that tells the function if the .zip file needs to be unzip or it already is.

    Returns
    -------
    
    Four dataframes: fa (self explanatory), 
                     func (refers to functional data), 
                     gm (refers to grey matter),
                     full_dataset (the three previous datasets in one)
    """

    if unzip:
        unzip_data()

    # 1. Create a dictionary with the subjects ids as keys, since the subjects ids 
    # are stored in the clinic information dataset, we'll have to load this dataset 
    # first, the ids are stored in the "id" variable.

    clinical_data = pd.read_excel(path + "subject_clinical_data.xlsx")

    pacient_ids = clinical_data[clinical_data.controls_ms == 1].id
    control_ids = clinical_data[clinical_data.controls_ms == 0].id
    general_ids = clinical_data.loc[:, "id"]

    subjects_path = {}

    for subject in general_ids:
        subjects_path[subject] = {}

    # 2. Set the id as the index in the Subject Clinical Information dataset (clinical_data)

    clinical_data.set_index("id", drop=True, inplace=True)

    # 3. Store the Subject Neuroimaging Information in the dictionary

    missing_subjects = list() # Since there are some subjects in the folder that are not included in the subjects xlsx

    # In the following loop we'll changing arround two different folders (PATIENTS, CONTROLS), 
    # which have three addtional folders each (FA, FUNC, GM_networks). Once we get into each folder,
    # we'll look for csv files, which are linked to the subjects ids (there are 4 different subjects id
    # format: 063MSVIS, c034MSVIS, FIS_105, sFIS_06)

    for name_0 in ("PATIENTS", "CONTROLS"):
        for name_1 in ("FA", "FUNC", "GM_networks"):
            for file in os.listdir(path + name_0 + "/" + name_1):
                if ".csv" in file:
                    if re.search(r'[0-9]{3}[A-Z]{5}', file): # 063MSVIS type of subjects
                        sub_id = re.search(r'[0-9]{3}[A-Z]{5}', file).group()
                        if sub_id in subjects_path.keys():
                            subjects_path[sub_id][name_1] = path + name_0 + "/" + name_1 + "/" + file
                        else:
                            if sub_id not in missing_subjects:
                                missing_subjects.append(sub_id)

                    if re.search(r'c[0-9]{3}[A-Z]{5}', file): # c034MSVIS type of subjects
                        sub_id = re.search(r'c[0-9]{3}[A-Z]{5}', file).group()
                        if sub_id in subjects_path.keys():
                            subjects_path[sub_id][name_1] = path + name_0 + "/" + name_1 + "/" + file
                        else:
                            if sub_id not in missing_subjects:
                                missing_subjects.append(sub_id)

                    if re.search(r'FIS_[0-9]{3}', file): # FIS_105 type of subjects
                        sub_id = re.search(r'FIS_[0-9]{3}', file).group()
                        if sub_id in subjects_path.keys():
                            subjects_path[sub_id][name_1] = path + name_0 + "/" + name_1 + "/" + file
                        else:
                            if sub_id not in missing_subjects:
                                missing_subjects.append(sub_id)

                    if re.search(r'sFIS_[0-9]{2}', file): # sFIS_06 type of subjects
                        sub_id = re.search(r'sFIS_[0-9]{2}', file).group()
                        if sub_id in subjects_path.keys():
                            subjects_path[sub_id][name_1] = path + name_0 + "/" + name_1 + "/" + file
                        else:
                            if sub_id not in missing_subjects:
                                missing_subjects.append(sub_id)

    # Once we have our path information, let's load it to a multidimensional array.

    neuro_data = np.zeros((165, 3, 76, 76), dtype=np.float32)

    for i,sub in enumerate(subjects_path.keys()):
        for j,name in enumerate(("FA", "FUNC", "GM_networks")):
            neuro_data[i, j, :, :] = pd.read_csv(subjects_path[sub][name], sep=',',header=None)

    # Now that we have our data into a 4d array, let's transform it to DFs

    # First of all, let's load the region of the brain linked to every index in the matrices

    matrix_index = list()

    with open(path + "mindboggle_ROIs.txt") as f:
        for line in f.readlines():
            matrix_index.append(line.split()[1])

    matrix_index = matrix_index[1:] # Skip the first row (title)

    # Replicate a matrix with 76 rows and 76 columns containing the names combinations

    matrix_names = list()

    for i in range(len(matrix_index)):
        matrix_names.append([])
        for j in range(len(matrix_index)):
            matrix_names[i].append(matrix_index[i] + "/" + matrix_index[j])

    # Apply a mask to the names matrix, keeping the upper triangle

    matrix_names = np.array(matrix_names)
    matrix_mask_names = matrix_names[np.triu_indices(matrix_names.shape[0])]

    matrix_mask_names_fa = ["fa-" + name for name in matrix_mask_names]
    matrix_mask_names_func = ["func-" + name for name in matrix_mask_names]
    matrix_mask_names_gm = ["gm-" + name for name in matrix_mask_names]

    # Apply a mask to every subject of the study

    masked_neuro_data = np.zeros((neuro_data.shape[0], neuro_data.shape[1], 2926), dtype=np.float32)

    for i in range(3):
        for j,subject in enumerate(neuro_data[:, i, :, :]):
            aux = neuro_data[j, i, :, :].copy()
            masked_neuro_data[j, i, :] = aux[np.triu_indices(aux.shape[0])]

    # Concert the data from numpy arrays to pandas dataframes, using the subject id as index and the region names of the brain as columns

    fa_data = pd.DataFrame(masked_neuro_data[:, 0, :], columns=matrix_mask_names_fa, index=list(subjects_path))
    func_data = pd.DataFrame(masked_neuro_data[:, 1, :], columns=matrix_mask_names_func, index=list(subjects_path))
    gm_data = pd.DataFrame(masked_neuro_data[:, 2, :], columns=matrix_mask_names_gm, index=list(subjects_path))

    # Finally, let's delete the main diagonal of the matrices because it doesn't add relevant information

    main_diagonal = [0]

    for i in range(75):
        aux = main_diagonal[i]
        main_diagonal.append(aux +  76 - i)

    useful_cols = [num for num in range(2926) if num not in main_diagonal]

    fa_data = fa_data.iloc[:, useful_cols]
    func_data = func_data.iloc[:, useful_cols]
    gm_data = gm_data.iloc[:, useful_cols]

    # Concat Clinical Data

    full_dataset = pd.concat([fa_data, func_data, gm_data, clinical_data.loc[:, ["age", "sex", "dd", "edss", "controls_ms"]]], join="inner", axis=1)
    fa_data = pd.concat([fa_data, clinical_data.loc[:, ["age", "sex", "dd", "edss", "controls_ms"]]], join="inner", axis=1)
    func_data = pd.concat([func_data, clinical_data.loc[:, ["age", "sex", "dd", "edss", "controls_ms"]]], join="inner", axis=1)
    gm_data = pd.concat([gm_data, clinical_data.loc[:, ["age", "sex", "dd", "edss", "controls_ms"]]], join="inner", axis=1)

    return full_dataset, fa_data, func_data, gm_data