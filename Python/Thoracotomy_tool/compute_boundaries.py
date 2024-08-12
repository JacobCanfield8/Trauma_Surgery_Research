import joblib
import numpy as np
import os
from itertools import combinations
from sklearn.base import BaseEstimator
import warnings

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but")

def compute_decision_boundary(model: BaseEstimator, feature_i: str, feature_j: str, feature_ranges: dict, all_features: list):
    x_min, x_max = feature_ranges[feature_i]
    y_min, y_max = feature_ranges[feature_j]
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Create a grid with default values for other features
    num_features = len(all_features)
    grid_data = np.zeros((grid_points.shape[0], num_features))
    
    # Place grid points in the correct columns
    index_i = all_features.index(feature_i)
    index_j = all_features.index(feature_j)
    grid_data[:, index_i] = grid_points[:, 0]
    grid_data[:, index_j] = grid_points[:, 1]

    Z = model.predict(grid_data)
    Z = Z.reshape(xx.shape)
    return xx, yy, Z

def generate_feature_combinations(features):
    return [list(comb) for i in range(2, len(features) + 1) for comb in combinations(features, i)]

def main():
    base_path = '/Users/JakeCanfield/Documents/Trauma_Surgery_Research/Python/Thoracotomy_tool/models_'
    
    model_types = ['xgb', 'logistic_regression']
    penalties = ['l1', 'l2', 'elasticnet']
    versions = ['v1.0', 'v2.0']
    data_types = {
        'EMS': ['SEX', 'AGEYEARS', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE', 'EMSTOTALGCS', 'PREHOSPITALCARDIACARREST', 'TRAUMATYPE'],
        'ED': ['SEX', 'AGEYEARS', 'SBP', 'PULSERATE', 'RESPIRATORYRATE', 'TOTALGCS', 'PREHOSPITALCARDIACARREST', 'TRAUMATYPE', 'TEMPERATURE'],
        'EMS_ED': ['SEX', 'AGEYEARS', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE', 'EMSTOTALGCS', 'SBP', 'PULSERATE', 'RESPIRATORYRATE', 'TOTALGCS', 'TEMPERATURE', 'PREHOSPITALCARDIACARREST', 'TRAUMATYPE', 'MECHANISM']
    }

    feature_ranges = {
        'SBP': (0, 300),
        'EMSSBP': (0, 300),
        'PULSERATE': (0, 300),
        'EMSPULSERATE': (0, 300),
        'RESPIRATORYRATE': (0, 60),
        'EMSRESPIRATORYRATE': (0, 60),
        'TOTALGCS': (3, 15),
        'EMSTOTALGCS': (3, 15),
        'TEMPERATURE': (30, 45),
        'AGEYEARS': (0, 100),
        'SEX': (1, 2),
        'TRAUMATYPE': (1, 4),
        'PREHOSPITALCARDIACARREST': (1, 2),
        'MECHANISM': (0, 23)
    }

    decision_boundaries = {}

    for version in versions:
        for data_type, features in data_types.items():
            feature_combinations = generate_feature_combinations(features)
            for comb in feature_combinations:
                ordered_features = '_'.join(comb)
                if 'xgb' in model_types:
                    model_filename = f"{base_path}{version}/xgb_model_{data_type}_{ordered_features}.pkl"
                    if os.path.exists(model_filename):
                        model = joblib.load(model_filename)
                        for i, j in combinations(comb, 2):
                            print(f"Computing decision boundary for {version}, {data_type}, xgb, features {i} and {j}")
                            xx, yy, Z = compute_decision_boundary(model, i, j, feature_ranges, comb)
                            decision_boundaries[(version, data_type, 'xgb', i, j)] = {'xx': xx, 'yy': yy, 'Z': Z}

                for penalty in penalties:
                    model_filename = f"{base_path}{version}/logistic_regression_{penalty}_model_{data_type}_{ordered_features}.pkl"
                    if os.path.exists(model_filename):
                        model = joblib.load(model_filename)
                        for i, j in combinations(comb, 2):
                            print(f"Computing decision boundary for {version}, {data_type}, logistic_regression_{penalty}, features {i} and {j}")
                            xx, yy, Z = compute_decision_boundary(model, i, j, feature_ranges, comb)
                            decision_boundaries[(version, data_type, f'logistic_regression_{penalty}', i, j)] = {'xx': xx, 'yy': yy, 'Z': Z}

    # Save the decision boundaries
    boundaries_filename = 'decision_boundaries.pkl'
    joblib.dump(decision_boundaries, boundaries_filename)
    print(f"Decision boundaries saved to {boundaries_filename}")

if __name__ == '__main__':
    main()
