import numpy as np
import pandas as pd
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib

data_fp = '/Users/JakeCanfield/Documents/Trauma_Surgery_Research/Data/Raw_data/PUF AY %i/CSV/'
years = range(2017, 2023)
thoracotomy_codes = ['0WJB0ZZ', '0WJ90ZZ', '02JA0ZZ', '02JY0ZZ', '0BJL0ZZ', '0BJK0ZZ', '0BJQ0ZZ','0WJ80ZZ', '0WJC0ZZ', '0WJD0ZZ', '02VW0CZ', '02QA0ZZ', '3E080GC']
sternotomy_codes = ['0P800ZZ']

mechanism_code_dict = {1:'Cut/pierce', 2:'Drowning/submersion', 3:'Fall', 4:'Fire/flame', 5:'Hot object/substance', 6:'Firearm', 7:'Machinery', 8:'MVT Occupant', 9:'MVT Motorcyclist', 10:'MVT Pedal cyclist', 11:'MVT Pedestrian', 12:'MVT Unspecified', 13:'MVT Other', 14:'Pedal cyclist, other', 15:'Pedestrian, other', 16:'Transport, other', 17:'Natural/environmental,  Bites and stings', 18:'Natural/environmental,  Other', 19:'Overexertion', 20:'Poisoning', 21:'Struck by, against', 22:'Suffocation', 23:'Other specified and classifiable', 24:'Other specified, not elsewhere classifiable', 25:'Unspecified', 26:'Adverse effects, medical care', 27:'Adverse effects, drugs'} # As noted in PUF dictionary
trauma_type_code_dict = {1:'Blunt', 2:'Penetrating', 3:'Burn', 4:'Other/unspecified', 9:'Activity Code - Not Valid as a Primary E-Code'} # As noted in PUF Dictionary
sex_code_dict = {1:'Male', 2:'Female', 3:'Unknown'}
eddischarge_code_dict = {1: 'Floor bed (general admission, non-specialty unit bed)', 2: 'Observation unit (unit that provides < 24 hour stays)', 3: 'Telemetry/step-down unit (less acuity than ICU)', 4: 'Home with services', 5: 'Deceased/expired', 6: 'Other (jail, institutional care, mental health, etc.)', 7: 'Operating Room', 8: 'Intensive Care Unit (ICU)', 9: 'Home without services', 10: 'Left against medical advice', 11: 'Transferred to another hospital'}
hospdischarge_disposition_code_dict = {1: 'Discharged/Transferred to a short-term general hospital for inpatient care', 2: 'Discharged/Transferred to an Intermediate Care Facility (ICF)', 3: 'Discharged/Transferred to home under care of organized home health service', 4: 'Left against medical advice or discontinued care', 5: 'Deceased/Expired', 6: 'Discharged to home or self-care (routine discharge)', 7: 'Discharged/Transferred to Skilled Nursing Facility (SNF)', 8: 'Discharged/Transferred to hospice care', 10: 'Discharged/Transferred to court/law enforcement', 11: 'Discharged/Transferred to inpatient rehab or designated unit', 12: 'Discharged/Transferred to Long Term Care Hospital (LTCH)', 13: 'Discharged/Transferred to a psychiatric hospital or psychiatric distinct part unit of a hospital', 14: 'Discharged/Transferred to another type of institution not defined elsewhere'}
deathined_code_dict = {1:'Arrived with NO signs of life', 2:'Arrived with signs of life'}
prehospca_code_dict = {1:'Yes', 2:'No'}
transport_mode_code_dict = {1:'Ground Ambulance', 2:'Helicopter Ambulance', 3:'Fixed-wing Ambulance', 4:'Private/Public Vehicle/Walk-in', 5:'Police', 6:'Other'}

'''
SEX: Male = 1; Female = 2
EMSSBP: #
EMSPULSERATE: #
EMSRESPIRATORYRATE:  #
EMSTOTALGCS: #
PREHOSPITALCARDIACARREST: Yes = 1; No = 2
TRAUMATYPE: Blunt = 1; Penetrating = 2; Burn = 3; Other/unspecified = 4; Activity Code - Not Valid as a Primary E-Code
'''

cols17 = pd.read_csv(data_fp%(2017)+'PUF_TRAUMA.csv', nrows=1).columns.tolist()
cols19 = pd.read_csv(data_fp%(2017)+'PUF_TRAUMA.csv', nrows=1).columns.tolist()
cols17 = [x.upper() for x in cols17]
cols19 = [x.upper() for x in cols19]

common_cols = list(set(cols17) & set(cols19))

for year in years:
    
    if year in range(2017, 2019):
        TRAUMA_df = pd.read_csv(data_fp%year + 'PUF_TRAUMA.csv')
        ICDPROCEDURE_df = pd.read_csv(data_fp%year + 'PUF_ICDPROCEDURE.csv')
        ICDPROCEDURE_LOOKUP_df = pd.read_csv(data_fp%year + 'PUF_ICDPROCEDURE_LOOKUP.csv')
        ECODE_LOOKUP_df = pd.read_csv(data_fp%year + 'PUF_ECODE_LOOKUP.csv')
    elif year in range(2019, 2023):
        TRAUMA_df = pd.read_csv(data_fp%year + 'PUF_TRAUMA.csv')
        ICDPROCEDURE_df = pd.read_csv(data_fp%year + 'PUF_ICDPROCEDURE.csv')
        ICDPROCEDURE_LOOKUP_df = pd.read_csv(data_fp%year + 'PUF_ICDPROCEDURE_LOOKUP.csv')
        ECODE_LOOKUP_df = pd.read_csv(data_fp%year + 'PUF_ECODE_LOOKUP.csv')
    else:
        pass
    
    TRAUMA_df.columns = map(str.upper, TRAUMA_df.columns)
    ICDPROCEDURE_df.columns = map(str.upper, ICDPROCEDURE_df.columns)
    ICDPROCEDURE_LOOKUP_df.columns = map(str.upper, ICDPROCEDURE_LOOKUP_df.columns)
    ECODE_LOOKUP_df.columns = map(str.upper, ECODE_LOOKUP_df.columns)
    
    if year in range(2019, 2023):
        ICDPROCEDURE_df['PROCEDUREMINS'] = ICDPROCEDURE_df['HOSPITALPROCEDURESTARTHRS']*60
        ICDPROCEDURE_df['PROCEDUREDAYS'] = ICDPROCEDURE_df['HOSPITALPROCEDURESTARTDAYS']
        TRAUMA_df['HMRRHGCTRLSURGMINS'] = TRAUMA_df['HMRRHGCTRLSURGHRS']*60
    else:
        pass
    
    ICDPROCEDURE_df = ICDPROCEDURE_df.loc[:, ['INC_KEY', 'ICDPROCEDURECODE', 'PROCEDUREMINS', 'PROCEDUREDAYS']]
    
    ECODE_LOOKUP_df = ECODE_LOOKUP_df.loc[:, ['ECODE', 'ECODE_DESC', 'MECHANISM', 'TRAUMATYPE']]
    
    trauma_cols = np.unique(['INC_KEY', 'PRIMARYECODEICD10', 'AGEYEARS', 'SEX', 'EDDISCHARGEDISPOSITION', 'HOSPDISCHARGEDISPOSITION', 'HMRRHGCTRLSURGMINS'] + common_cols)
    TRAUMA_df = TRAUMA_df.loc[:, trauma_cols]
    TRAUMA_df = TRAUMA_df.loc[TRAUMA_df['HMRRHGCTRLSURGTYPE'] == 3.0]
    
    ICDPROCEDURE_df = ICDPROCEDURE_df.loc[ICDPROCEDURE_df['INC_KEY'].isin(np.unique(TRAUMA_df['INC_KEY']))]
    
    ICDPROCEDURE_LOOKUP_df = ICDPROCEDURE_LOOKUP_df.loc[:, ['ICDPROCEDURECODE', 'ICDPROCEDURECODE_DESC']]
    ECODE_LOOKUP_df = ECODE_LOOKUP_df.loc[:, ['ECODE', 'ECODE_DESC', 'MECHANISM', 'TRAUMATYPE']]
    
    ecode_dict = dict(zip(ECODE_LOOKUP_df['ECODE'], ECODE_LOOKUP_df['ECODE_DESC'])) # create dictionary
    TRAUMA_df['PRIMARYECODEICD10'] = TRAUMA_df['PRIMARYECODEICD10'].replace(ecode_dict) # implement dictionary
    
    mechanism_dict = dict(zip(ECODE_LOOKUP_df['ECODE_DESC'], ECODE_LOOKUP_df['MECHANISM'])) # create dictionary
    TRAUMA_df['MECHANISM'] = TRAUMA_df['PRIMARYECODEICD10'].map(mechanism_dict) # implement dictionary
    
    traumatype_dict = dict(zip(ECODE_LOOKUP_df['ECODE_DESC'], ECODE_LOOKUP_df['TRAUMATYPE'])) # create dictionary
    TRAUMA_df['TRAUMATYPE'] = TRAUMA_df['PRIMARYECODEICD10'].map(traumatype_dict) # implement dictionary
    
    trauma_cols = np.unique(['INC_KEY', 'PRIMARYECODEICD10', 'AGEYEARS', 'SEX', 'EDDISCHARGEDISPOSITION', 'HOSPDISCHARGEDISPOSITION', 'TRAUMATYPE', 'MECHANISM'] + common_cols)
    TRAUMA_df = TRAUMA_df.loc[:, trauma_cols]
    TRAUMA_df = TRAUMA_df.loc[TRAUMA_df['INC_KEY'].isin(np.unique(ICDPROCEDURE_df['INC_KEY']))]
    
    TRAUMA_df.to_csv('/Users/JakeCanfield/Documents/Trauma_Surgery_Research/data/Combined_data/MLTRAUMA_df_%i.csv'%year, index=False)
    ICDPROCEDURE_df.to_csv('/Users/JakeCanfield/Documents/Trauma_Surgery_Research/data/Combined_data/MLICDPROCEDURE_df_%i.csv'%year, index=False)

    # Free up memory
    del TRAUMA_df, ICDPROCEDURE_df, ICDPROCEDURE_LOOKUP_df, ECODE_LOOKUP_df
    
TRAUMA_fp = '/Users/JakeCanfield/Documents/Trauma_Surgery_Research/data/Combined_data/MLTRAUMA_df_%i.csv'
ICDPROCEDURE_fp = '/Users/JakeCanfield/Documents/Trauma_Surgery_Research/data/Combined_data/MLICDPROCEDURE_df_%i.csv'

years = range(2017, 2023)

# Preload column names
TRAUMA_cols = pd.read_csv(TRAUMA_fp % 2017, nrows=1).columns.tolist()
ICDPROCEDURE_cols = pd.read_csv(ICDPROCEDURE_fp % 2017, nrows=1).columns.tolist()

# Collect DataFrames in lists
trauma_dfs = []
icdprocedure_dfs = []

for year in years:
    trauma_df = pd.read_csv(TRAUMA_fp % year, usecols=TRAUMA_cols)
    trauma_df['Year'] = year
    icdprocedure_df = pd.read_csv(ICDPROCEDURE_fp % year, usecols=ICDPROCEDURE_cols)
    icdprocedure_df['Year'] = year
    trauma_dfs.append(trauma_df)
    icdprocedure_dfs.append(icdprocedure_df)

# Concatenate DataFrames once at the end
TRAUMA_all_df = pd.concat(trauma_dfs, ignore_index=True)
ICDPROCEDURE_all_df = pd.concat(icdprocedure_dfs, ignore_index=True)

TRAUMA_all_df = TRAUMA_all_df.dropna(subset=['HMRRHGCTRLSURGMINS'])
TRAUMA_all_df = TRAUMA_all_df.dropna(subset=['AGEYEARS'])
TRAUMA_all_df['HMRRHGCTRLSURGMINS'] = TRAUMA_all_df['HMRRHGCTRLSURGMINS'].round().astype(int)
TRAUMA_all_df = TRAUMA_all_df.loc[(TRAUMA_all_df['HMRRHGCTRLSURGMINS'] <= 15) & (TRAUMA_all_df['HMRRHGCTRLSURGDAYS'] <= 1.0)]
ICDPROCEDURE_all_df = ICDPROCEDURE_all_df.loc[ICDPROCEDURE_all_df['INC_KEY'].isin(np.unique(TRAUMA_all_df['INC_KEY']))]

# Step 1: Drop rows that have NaN in both columns
TRAUMA_all_df = TRAUMA_all_df.dropna(subset=['EDDISCHARGEDISPOSITION', 'HOSPDISCHARGEDISPOSITION'], how='all')

# Step 2: Replace NaN values in one column with the value from the other column
TRAUMA_all_df['EDDISCHARGEDISPOSITION'] = TRAUMA_all_df['EDDISCHARGEDISPOSITION'].combine_first(TRAUMA_all_df['HOSPDISCHARGEDISPOSITION'])
TRAUMA_all_df['HOSPDISCHARGEDISPOSITION'] = TRAUMA_all_df['HOSPDISCHARGEDISPOSITION'].combine_first(TRAUMA_all_df['EDDISCHARGEDISPOSITION'])

# Step 3: Create the 'DECEASED' column
TRAUMA_all_df['DECEASED'] = 0  # Initialize with 0 (Survived)
TRAUMA_all_df.loc[(TRAUMA_all_df['EDDISCHARGEDISPOSITION'] == 5.0) | 
                  (TRAUMA_all_df['HOSPDISCHARGEDISPOSITION'] == 5.0), 'DECEASED'] = 1

# Step 4: Drop the 'EDDISCHARGEDISPOSITION' and 'HOSPDISCHARGEDISPOSITION' columns
TRAUMA_all_df = TRAUMA_all_df.drop(columns=['EDDISCHARGEDISPOSITION', 'HOSPDISCHARGEDISPOSITION', 'INC_KEY'])

# Step 5: Select rows of interest
input_list = ['SEX', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE', 'EMSTOTALGCS', 'PREHOSPITALCARDIACARREST', 'TRAUMATYPE']
required_list = ['DECEASED']
TRAUMA_all_df = TRAUMA_all_df.loc[:, input_list + required_list]

TRAUMA_all_df = TRAUMA_all_df.dropna(subset=input_list, how='any')

# Step 8: Remove columns ending in _BIU
TRAUMA_all_df = TRAUMA_all_df.drop(TRAUMA_all_df.filter(regex='_BIU$').columns, axis=1)

# Step 9: Convert strings to numbers

# Identify columns with string values
string_columns = TRAUMA_all_df.select_dtypes(include='object').columns

# Initialize a dictionary to store mapping dictionaries
mapping_dicts = {}

# Convert each unique string to an integer and create mapping dictionaries
for col in string_columns:
    unique_strings = TRAUMA_all_df[col].unique()
    string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
    int_to_string = {idx: string for string, idx in string_to_int.items()}
    
    # Store the mapping dictionaries
    mapping_dicts[col] = {
        'string_to_int': string_to_int,
        'int_to_string': int_to_string
    }
    
    # Map the strings to integers in the DataFrame
    TRAUMA_all_df[col] = TRAUMA_all_df[col].map(string_to_int)

# Free up memory
del trauma_dfs, icdprocedure_dfs

def make_splits(input_df, columns=None, y_column='DECEASED', train_val_test_split=0.2, val_test_split=0.5, random_state=8):
    
    # Step 1: Select specific input columns to use, otherwise use all columns (do not include y_column in columns if not using all columns, it gets added if you select a subset of columns)
    if columns == None:
        pass
    else:
        input_df = input_df.loc[:, columns + [y_column]]
    
    # Step 2: Split the data into 80% training and 20% remaining
    train_df, remaining_df = train_test_split(
        input_df, 
        test_size=train_val_test_split, 
        stratify=input_df[y_column], 
        random_state=random_state)
    
    # Step 3: Split the remaining data into 50% validation and 50% test
    val_df, test_df = train_test_split(
        remaining_df, 
        test_size=val_test_split, 
        stratify=remaining_df[y_column], 
        random_state=random_state)
    
    # Step 4: Split into X and y
    X_train = train_df.drop(columns=[y_column])
    y_train = train_df[y_column]
    
    X_val = val_df.drop(columns=[y_column])
    y_val = val_df[y_column]
    
    X_test = test_df.drop(columns=[y_column])
    y_test = test_df[y_column]

    return X_train, y_train, X_val, y_val, X_test, y_test

def GBM_classifier(X_train, y_train, X_val, y_val, X_test, y_test, learning_rate=None, n_estimators=None, print_out=False, XGBM=False):
    """
    Train and evaluate a Gradient Boost Machine (GBM) classifier with a given n_estimators.
    
    Parameters:
    n_estimator: 
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target.
    
    Returns:
    model (object): Trained model.
    y_train_pred (np.ndarray): Predicted target values for training set.
    y_val_pred (np.ndarray): Predicted target values for validation set.
    y_test_pred (np.ndarray): Predicted target values for test set.
    y_train_proba (np.ndarray): Predicted probabilities for training set.
    y_val_proba (np.ndarray): Predicted probabilities for validation set.
    y_test_proba (np.ndarray): Predicted probabilities for test set.
    roc_auc_val (float): AUROC for validation set.
    roc_auc_test (float): AUROC for test set.
    """
    if XGBM == False:
        model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    elif XGBM == True:
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7],
            'scale_pos_weight': [1, y_train.value_counts()[0] / y_train.value_counts()[1]]
        }
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        y_test_pred = best_model.predict(X_test)
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        # Save the best model
        joblib.dump(best_model, 'xgb_model.pkl')
    else:
        raise ValueError("Invalid model choice")

    # Calculate AUROC
    roc_auc_val = auc(*roc_curve(y_val, y_val_proba)[:2])
    roc_auc_test = auc(*roc_curve(y_test, y_test_proba)[:2])
    
    return model, y_train_pred, y_val_pred, y_test_pred, y_train_proba, y_val_proba, y_test_proba, roc_auc_val, roc_auc_test

# Split the data
X_train, y_train, X_val, y_val, X_test, y_test = make_splits(TRAUMA_all_df, y_column='DECEASED')

# Train the model and get predictions and metrics
model, y_train_pred, y_val_pred, y_test_pred, y_train_proba, y_val_proba, y_test_proba, roc_auc_val, roc_auc_test = GBM_classifier(
    X_train, y_train, X_val, y_val, X_test, y_test, XGBM=True, learning_rate=0.1, n_estimators=100)