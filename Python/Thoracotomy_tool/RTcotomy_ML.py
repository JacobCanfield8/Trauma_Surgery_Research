import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

data_fp = '/Users/JakeCanfield/Documents/Trauma_Surgery_Research/Data/Vitals/vitals_%i_mins_df_%i.csv'
years = range(2007, 2021, 1)
cutofftime = 20

trauma_dfs = []
for year in years:
    trauma_df = pd.read_csv(data_fp % (cutofftime, year))
    trauma_df['Year'] = year
    trauma_dfs.append(trauma_df)

# Concatenate DataFrames once at the end
TRAUMA_all_df = pd.concat(trauma_dfs, ignore_index=True)

TRAUMA_all_df = TRAUMA_all_df.dropna(subset=['AGEYEARS'])

# Step 1: Drop rows that have NaN in both columns
TRAUMA_all_df = TRAUMA_all_df.dropna(subset=['EDDISCHARGEDISPOSITION', 'HOSPDISCHARGEDISPOSITION'], how='all')

# Step 2: Replace NaN values in one column with the value from the other column
TRAUMA_all_df['EDDISCHARGEDISPOSITION'] = TRAUMA_all_df['EDDISCHARGEDISPOSITION'].combine_first(TRAUMA_all_df['HOSPDISCHARGEDISPOSITION'])
TRAUMA_all_df['HOSPDISCHARGEDISPOSITION'] = TRAUMA_all_df['HOSPDISCHARGEDISPOSITION'].combine_first(TRAUMA_all_df['EDDISCHARGEDISPOSITION'])

# Step 3: Create the 'DECEASED' column
TRAUMA_all_df['DECEASED'] = 0  # Initialize with 0 (Survived)
TRAUMA_all_df.loc[(TRAUMA_all_df['EDDISCHARGEDISPOSITION'].isin(['Expired', 'Deceased/Expired', 'Deceased/expired'])) | 
                  (TRAUMA_all_df['HOSPDISCHARGEDISPOSITION'].isin(['Expired', 'Deceased/Expired', 'Deceased/expired'])), 'DECEASED'] = 1

# Step 4: Drop the 'EDDISCHARGEDISPOSITION' and 'HOSPDISCHARGEDISPOSITION' columns
TRAUMA_all_df = TRAUMA_all_df.drop(columns=['EDDISCHARGEDISPOSITION', 'HOSPDISCHARGEDISPOSITION', 'INC_KEY'])

# Step 5: Select columns of interest
EMS_list = ['SEX', 'AGEYEARS', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE', 'EMSTOTALGCS', 'PREHOSPITALCARDIACARREST', 'TRAUMATYPE', 'MECHANISM']
ED_list = ['SEX', 'AGEYEARS', 'SBP', 'PULSERATE', 'RESPIRATORYRATE', 'TOTALGCS', 'PREHOSPITALCARDIACARREST', 'TRAUMATYPE', 'MECHANISM', 'TEMPERATURE']
EMS_ED_list = ['SEX', 'AGEYEARS', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE', 'EMSTOTALGCS', 'SBP', 'PULSERATE', 'RESPIRATORYRATE', 'TOTALGCS', 'TEMPERATURE', 'PREHOSPITALCARDIACARREST', 'TRAUMATYPE', 'MECHANISM']

feature_sets = {
    'EMS': EMS_list,
    'ED': ED_list,
    'EMS_ED': EMS_ED_list
}

# Initialize a dictionary to store mapping dictionaries
mapping_dicts = {}

# Convert each unique string to an integer and create mapping dictionaries
string_columns = TRAUMA_all_df.select_dtypes(include='object').columns
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
    
def clean_data(X, y):
    before_drop = len(X)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index]
    after_drop = len(X)
    print(f"Dropped {before_drop - after_drop} rows due to NaN or infinity values.")
    return X, y

def make_splits(input_df, columns, y_column='DECEASED', train_val_test_split=0.2, val_test_split=0.5, random_state=8):
    input_df = input_df.loc[:, columns + [y_column]]
    
    train_df, remaining_df = train_test_split(
        input_df, 
        test_size=train_val_test_split, 
        stratify=input_df[y_column], 
        random_state=random_state)
    
    val_df, test_df = train_test_split(
        remaining_df, 
        test_size=val_test_split, 
        stratify=remaining_df[y_column], 
        random_state=random_state)
    
    X_train = train_df.drop(columns=[y_column])
    y_train = train_df[y_column]
    
    X_val = val_df.drop(columns=[y_column])
    y_val = val_df[y_column]
    
    X_test = test_df.drop(columns=[y_column])
    y_test = test_df[y_column]

    X_train, y_train = clean_data(X_train, y_train)
    X_val, y_val = clean_data(X_val, y_val)
    X_test, y_test = clean_data(X_test, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_and_save_models(X_train, y_train, X_val, y_val, X_test, y_test, data_type, features):
    results = []
    
    penalties = ['l1', 'l2', 'elasticnet']
    param_grid = {
        'C': [1/0.00001, 1/0.0001, 1/0.001, 1/0.01, 1/0.1, 1/1.0]
    }
    
    for penalty in penalties:
        if penalty == 'elasticnet':
            param_grid['l1_ratio'] = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            solver = 'saga'
        else:
            solver = 'liblinear'
            param_grid.pop('l1_ratio', None)
        
        model = LogisticRegression(max_iter=10000, solver=solver, penalty=penalty)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=2, n_jobs=-1)
        
        print(f"Training Logistic Regression {penalty} for {data_type} with features {features}")
        print(f"y_train distribution: {y_train.value_counts()}")
        
        if len(np.unique(y_train)) < 2:
            print(f"Skipping training for {data_type} with features {features} due to insufficient class diversity.")
            continue
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        y_test_pred = best_model.predict(X_test)
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        samples_used = len(X_train) + len(X_val) + len(X_test)
        
        results.append([data_type, penalty, ','.join(features), accuracy, roc_auc, samples_used])
        joblib.dump(best_model, f'models/logistic_regression_{penalty}_model_{data_type}_{"_".join(features)}.pkl')
    
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7]
    }
    
    print(f"Training XGBoost for {data_type} with features {features}")
    print(f"y_train distribution: {y_train.value_counts()}")
    
    if len(np.unique(y_train)) >= 2:
        y_train_values = y_train.value_counts()
        print(f"y_train value counts: {y_train_values}")
        if 0 in y_train_values and 1 in y_train_values:
            param_grid['scale_pos_weight'] = [1, y_train_values[0] / y_train_values[1]]
        else:
            print(f"Skipping XGBoost training for {data_type} with features {features} due to insufficient class diversity.")
            return results

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
        
        accuracy = accuracy_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        samples_used = len(X_train) + len(X_val) + len(X_test)
        
        results.append([data_type, 'xgb', ','.join(features), accuracy, roc_auc, samples_used])
        joblib.dump(best_model, f'models/xgb_model_{data_type}_{"_".join(features)}.pkl')
    
    return results

results = []
for data_type, features in feature_sets.items():
    for i in range(2, len(features) + 1):  # Ensure at least 2 features are used
        for subset in combinations(features, i):
            X_train, y_train, X_val, y_val, X_test, y_test = make_splits(TRAUMA_all_df, columns=list(subset))
            print(f"Training models for {data_type} with features {subset}")
            print(f"y_train distribution: {y_train.value_counts()}")
            if len(np.unique(y_train)) < 2:
                print(f"Skipping training for {data_type} with features {subset} due to insufficient class diversity.")
                continue
            results.extend(train_and_save_models(X_train, y_train, X_val, y_val, X_test, y_test, data_type, list(subset)))

results_df = pd.DataFrame(results, columns=['Data Type', 'Model Type', 'Features', 'Accuracy', 'AUROC', 'Samples Used'])
results_df.to_csv('models/model_performance_metrics.csv', index=False)
