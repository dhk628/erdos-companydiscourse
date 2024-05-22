import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pandas.api.types import CategoricalDtype
import joblib
import os
from statistics import fmean

N_SPLITS = 10
KFOLD_SEED = 0


def balance_and_fit(indep, dep, model, balancing=None):
    if balancing:
        indep_resampled, dep_resampled = balancing.fit_resample(indep, dep)
    else:
        indep_resampled = indep
        dep_resampled = dep
    return model.fit(indep_resampled, dep_resampled)


def save_models_kfold(indep, dep, model, balancing=None):
    kfold = StratifiedKFold(n_splits=N_SPLITS,
                            shuffle=True,
                            random_state=KFOLD_SEED)

    for i, (train_index, test_index) in enumerate(kfold.split(indep, dep)):
        print('Starting ' + str(i) + 'th validation...')
        path = "Models/" + type(model).__name__ + "_val_" + str(i) + '_' + type(balancing).__name__ + ".pkl"
        if os.path.exists(path):
            print(path + ' already exists.')
        else:
            # Get the kfold training data
            indep_train = indep[train_index, :]
            dep_train = dep[train_index]

            # # Get the validation data
            # X_test = X[test_index, :]
            # y_test = y[test_index]

            fit_model = balance_and_fit(indep_train, dep_train, model, balancing)
            joblib.dump(fit_model, path)
            print('Saved ' + str(i) + 'th validation.')


def load_models(model_name, balancing_name=None):
    models = []
    for i in range(N_SPLITS):
        path = "Models/" + model_name + "_val_" + str(i) + '_' + balancing_name + ".pkl"
        if os.path.exists(path):
            models.append(joblib.load("Models/" + model_name + "_val_" + str(i) + '_' + balancing_name + ".pkl"))
        else:
            raise Exception(path + " does not exist.")

    return models


def evaluate_regression(indep, dep, model_name, balancing_name):
    kfold = StratifiedKFold(n_splits=N_SPLITS,
                            shuffle=True,
                            random_state=KFOLD_SEED)

    models_list = load_models(model_name, balancing_name)
    evaluation_path = "Models/Evaluation/" + model_name + '_' + balancing_name + ".csv"

    if os.path.exists(evaluation_path):
        print('Evaluation already exists.')
    else:
        rmses, mses, maes, r_rmses, r_mses, r_maes = [], [], [], [], [], []

        for i, (train_index, test_index) in enumerate(kfold.split(indep, dep)):
            model = models_list[i]

            # Get the validation data
            indep_test = indep[test_index, :]
            dep_test = dep[test_index]

            predictions = model.predict(indep_test)

            rmses.append(root_mean_squared_error(dep_test, predictions))
            mses.append(mean_squared_error(dep_test, predictions))
            maes.append(mean_absolute_error(dep_test, predictions))

            r_predictions = np.around(predictions)
            r_rmses.append(root_mean_squared_error(dep_test, r_predictions))
            r_mses.append(mean_squared_error(dep_test, r_predictions))
            r_maes.append(mean_absolute_error(dep_test, r_predictions))

        rmses.append(fmean(rmses))
        mses.append(fmean(mses))
        maes.append(fmean(maes))
        r_rmses.append(fmean(r_rmses))
        r_mses.append(fmean(r_mses))
        r_maes.append(fmean(r_maes))

        data_dict = {'RMSE': rmses,
                     'MSE': mses,
                     'MAE': maes,
                     'Rounded RMSE': r_rmses,
                     'Rounded MSE': r_mses,
                     'Rounded MAE': r_maes}

        data = pd.DataFrame(data_dict)
        data.rename(index={N_SPLITS: 'AVG'}, inplace=True)
        data.to_csv(evaluation_path)


df = pd.read_parquet('Data/Vectorized/costco_2021_reviews_filtered_vectorized_master_en_float32_nodup.parquet')

# rating_counts = df['rating'].value_counts()
# print('Number of ratings:')
# print(rating_counts)

X = np.array(df['vector'].tolist())
y = np.array(df['rating'].tolist())

# Train-test split
X_0, X_final_test, y_0, y_final_test = train_test_split(X, y,
                                                        shuffle=True,
                                                        random_state=123,
                                                        stratify=y,
                                                        test_size=0.2)

# Example usage
# save_models_kfold(X_0, y_0, LinearRegression(), RandomUnderSampler(random_state=0))
# evaluate_regression(X_0, y_0, 'LinearRegression', 'RandomUnderSampler')
