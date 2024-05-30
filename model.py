import os
import numpy as np
import pandas as pd
from statistics import fmean
import pickle as pk
import joblib
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, mean_squared_error, root_mean_squared_error, mean_absolute_error, log_loss, accuracy_score, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids, CondensedNearestNeighbour, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pandas.api.types import CategoricalDtype


N_SPLITS = 11
RANDOM_SEED = 123
CLASS_NAMES = np.array([1, 2, 3, 4, 5])
# matplotlib.use('Agg')


def balance_and_fit(indep, dep, model, balancing=None, balancing2=None):
    if balancing and not balancing2:
        indep_resampled, dep_resampled = balancing.fit_resample(indep, dep)
    elif balancing and balancing2:
        indep_1, dep_1 = balancing.fit_resample(indep, dep)
        indep_resampled, dep_resampled = balancing2.fit_resample(indep_1, dep_1)
    else:
        indep_resampled = indep
        dep_resampled = dep
    return model.fit(indep_resampled, dep_resampled)


def save_models_kfold(indep, dep,
                      model, model_appendix='',
                      pca_dim=0,
                      balancing=None, balancing_appendix='',
                      balancing2=None, balancing2_appendix=''):
    kfold = StratifiedKFold(n_splits=N_SPLITS,
                            shuffle=True,
                            random_state=RANDOM_SEED)

    if model_appendix:
        model_appendix = '_' + model_appendix
    if pca_dim > 0:
        pca_name = '_PCA' + str(pca_dim)
    else:
        pca_name = ''
    if balancing_appendix:
        balancing_appendix = '_' + balancing_appendix
    if balancing2_appendix:
        balancing2_appendix = '_' + balancing2_appendix

    for i, (train_index, test_index) in enumerate(kfold.split(indep, dep)):
        print('Starting validation #' + str(i) + '...')
        if balancing2:
            path = "Models/" + type(model).__name__ + model_appendix + pca_name + '_' + type(balancing).__name__ + balancing_appendix + '_' + type(balancing2).__name__ + balancing2_appendix + "_val_" + str(i) + ".pkl"
        else:
            path = "Models/" + type(model).__name__ + model_appendix + pca_name + '_' + type(balancing).__name__ + balancing_appendix + "_val_" + str(i) + ".pkl"
        if os.path.exists(path):
            print(path + ' already exists.')
        else:
            # Get the kfold training data
            indep_train = indep[train_index, :]
            dep_train = dep[train_index]

            # # Get the validation data
            # X_test = X[test_index, :]
            # y_test = y[test_index]

            if pca_dim > 0:
                pca_path = 'Models/PCA/PCA' + "_val_" + str(i) + '.pkl'
                if os.path.exists(pca_path):
                    pca = pk.load(open(pca_path, 'rb'))
                    indep_train = pca.transform(indep_train)[:, :pca_dim]
                else:
                    pca = PCA(n_components=1024)
                    indep_train = pca.fit_transform(indep_train)
                    pk.dump(pca, open(pca_path, 'wb'))

            fit_model = balance_and_fit(indep_train, dep_train, model, balancing, balancing2)
            joblib.dump(fit_model, path)


# def save_models_kfold_pca(indep, dep, model, n_components, balancing=None, model_appendix=''):
#     kfold = StratifiedKFold(n_splits=N_SPLITS,
#                             shuffle=True,
#                             random_state=KFOLD_SEED)
#     rmses, mses, maes, r_rmses, r_mses, r_maes = [], [], [], [], [], []
#     evaluation_path = "Models/Evaluation/" + type(model).__name__ + "_pca_" + '_' + type(balancing).__name__ + ".csv"
#     for i, (train_index, test_index) in enumerate(kfold.split(indep, dep)):
#         print('Starting ' + str(i) + 'th validation...')
#         path = "Models/" + type(model).__name__ + "_pca_" + "_val_" + str(i) + '_' + type(balancing).__name__ + ".pkl"
#         if os.path.exists(path):
#             print(path + ' already exists.')
#         else:
#             # Get the kfold training data
#             indep_train = indep[train_index, :]
#             dep_train = dep[train_index]
#
#             # # Get the validation data
#             # X_test = X[test_index, :]
#             # y_test = y[test_index]
#
#             pca = PCA(n_components=n_components)
#             indep_train_reduced = pca.fit_transform(indep_train)
#
#             with open('Models/pca_' + str(i), 'wb') as handle:
#                 pkl.dump(pca, handle, protocol=pkl.HIGHEST_PROTOCOL)
#
#             fit_model = balance_and_fit(indep_train_reduced, dep_train, model, balancing)
#             joblib.dump(fit_model, path)
#             print('Saved ' + str(i) + 'th validation.')
#
#             # Get the validation data
#             indep_test = indep[test_index, :]
#             indep_test_reduced = pca.fit_transform(indep_test)
#             dep_test = dep[test_index]
#
#             predictions = model.predict(indep_test_reduced)
#
#             rmses.append(root_mean_squared_error(dep_test, predictions))
#             mses.append(mean_squared_error(dep_test, predictions))
#             maes.append(mean_absolute_error(dep_test, predictions))
#
#             r_predictions = np.around(predictions)
#             r_rmses.append(root_mean_squared_error(dep_test, r_predictions))
#             r_mses.append(mean_squared_error(dep_test, r_predictions))
#             r_maes.append(mean_absolute_error(dep_test, r_predictions))
#
#     rmses.append(fmean(rmses))
#     mses.append(fmean(mses))
#     maes.append(fmean(maes))
#     r_rmses.append(fmean(r_rmses))
#     r_mses.append(fmean(r_mses))
#     r_maes.append(fmean(r_maes))
#
#     data_dict = {'RMSE': rmses,
#                  'MSE': mses,
#                  'MAE': maes,
#                  'Rounded RMSE': r_rmses,
#                  'Rounded MSE': r_mses,
#                  'Rounded MAE': r_maes}
#
#     data = pd.DataFrame(data_dict)
#     data.rename(index={N_SPLITS: 'AVG'}, inplace=True)
#     data.to_csv(evaluation_path)


# Ordinal regression requires different syntax
# def save_ordreg_kfold(indep, dep, distr, method, balancing=None, balancing_2=None):
#     kfold = StratifiedKFold(n_splits=N_SPLITS,
#                             shuffle=True,
#                             random_state=KFOLD_SEED)
#
#     for i, (train_index, test_index) in enumerate(kfold.split(indep, dep)):
#         print('Starting ' + str(i) + 'th validation...')
#         if balancing_2:
#             path = "Models/" + 'OrderedRegression' + "_val_" + str(i) + '_' + type(balancing).__name__ + '_' + type(balancing_2).__name__ + ".pkl"
#         else:
#             path = "Models/" + 'OrderedRegression' + "_val_" + str(i) + '_' + type(balancing).__name__ + ".pkl"
#         if os.path.exists(path):
#             print(path + ' already exists.')
#         else:
#             # Get the kfold training data
#             indep_train = indep[train_index, :]
#             dep_train = dep[train_index]
#             if balancing:
#                 indep_resampled, dep_resampled = balancing.fit_resample(indep_train, dep_train)
#             else:
#                 indep_resampled = indep
#                 dep_resampled = dep
#
#             dep_resampled_ord = pd.Series(dep_resampled).astype(CategoricalDtype(ordered=True))
#             model = OrderedModel(dep_resampled_ord,
#                                  indep_resampled,
#                                  distr=distr)
#             fit_model = model.fit(method=method)
#             joblib.dump(fit_model, path)
#             print('Saved ' + str(i) + 'th validation.')


def load_models(model_name, balancing_name=None):
    models = []
    for i in range(N_SPLITS):
        path = "Models/" + model_name + '_' + balancing_name + "_val_" + str(i) + ".pkl"
        if os.path.exists(path):
            models.append(joblib.load(path))
        else:
            raise Exception(path + " does not exist.")

    return models


def evaluate_regression(indep, dep, model_name, balancing_name):
    kfold = StratifiedKFold(n_splits=N_SPLITS,
                            shuffle=True,
                            random_state=RANDOM_SEED)

    models_list = load_models(model_name, balancing_name)
    evaluation_path = "Models/Evaluation/" + model_name + '_' + balancing_name + ".csv"
    cm_norm_path = "Models/Evaluation/Confusion Matrix/" + model_name + '_' + balancing_name + '_normed' + ".png"

    if os.path.exists(evaluation_path) or os.path.exists(cm_norm_path):
        print('Evaluation or confusion matrices already exists.')
    else:
        rmses, mses, maes, r_rmses, r_mses, r_maes = [], [], [], [], [], []
        accuracies = []
        cms, cms_norm = [], []

        for i, (train_index, test_index) in enumerate(kfold.split(indep, dep)):
            print('Testing validation #' + str(i) + '...')
            model = models_list[i]

            # Get the validation data
            indep_test = indep[test_index, :]
            dep_test = dep[test_index]

            predictions = model.predict(indep_test)

            rmses.append(root_mean_squared_error(dep_test, predictions))
            mses.append(mean_squared_error(dep_test, predictions))
            maes.append(mean_absolute_error(dep_test, predictions))

            r_predictions = np.around(predictions).astype(int)
            for i in range(len(r_predictions)):
                if r_predictions[i] < 1:
                    r_predictions[i] = 1
                if r_predictions[i] > 5:
                    r_predictions[i] = 5
            r_rmses.append(root_mean_squared_error(dep_test, r_predictions))
            r_mses.append(mean_squared_error(dep_test, r_predictions))
            r_maes.append(mean_absolute_error(dep_test, r_predictions))

            accuracies.append(accuracy_score(dep_test, r_predictions))

            cms.append(confusion_matrix(dep_test, r_predictions, normalize=None))
            cms_norm.append(confusion_matrix(dep_test, r_predictions, normalize='true'))

        rmses.append(fmean(rmses))
        mses.append(fmean(mses))
        maes.append(fmean(maes))
        r_rmses.append(fmean(r_rmses))
        r_mses.append(fmean(r_mses))
        r_maes.append(fmean(r_maes))
        accuracies.append(fmean(accuracies))

        cms_norm_average = np.mean(np.array(cms_norm), axis=0)
        disp_norm = ConfusionMatrixDisplay(cms_norm_average, display_labels=CLASS_NAMES)
        disp_norm.plot().figure_.savefig(cm_norm_path)
        cms_norm_stacked = np.vstack(tuple([cms_norm_average] + cms_norm))
        cms_index = [[str(i), '', '', '', ''] for i in range(N_SPLITS)]
        cms_index_flattened = ['AVG', '', '', '', ''] + [item for row in cms_index for item in row]
        data_cms_norm = pd.DataFrame(cms_norm_stacked, index=cms_index_flattened,
                                     columns=['Predicted 1', 'Predicted 2', 'Predicted 3', 'Predicted 4',
                                              'Predicted 5'])
        data_cms_norm.insert(loc=0, column='',
                             value=['True 1', 'True 2', 'True 3', 'True 4', 'True 5'] * (N_SPLITS + 1))

        cms_average = np.mean(np.array(cms), axis=0)
        cms_stacked = np.vstack(tuple([cms_average] + cms))
        data_cms = pd.DataFrame(cms_stacked, index=cms_index_flattened,
                                columns=['Predicted 1', 'Predicted 2', 'Predicted 3', 'Predicted 4', 'Predicted 5'])
        data_cms.insert(loc=0, column='', value=['True 1', 'True 2', 'True 3', 'True 4', 'True 5'] * (N_SPLITS + 1))

        data_dict = {'RMSE': rmses,
                     'MSE': mses,
                     'MAE': maes,
                     'Rounded RMSE': r_rmses,
                     'Rounded MSE': r_mses,
                     'Rounded MAE': r_maes,
                     'Accuracy': accuracies}

        data = pd.DataFrame(data_dict)
        data.rename(index={N_SPLITS: 'AVG'}, inplace=True)
        data.loc[len(data)] = pd.Series(dtype='float64')
        data.rename(index={N_SPLITS + 1: ''}, inplace=True)
        data.to_csv(evaluation_path)

        data_cms_norm.to_csv(evaluation_path, mode='a')
        data_cms.to_csv(evaluation_path, mode='a')


def evaluate_classification(indep, dep, model_name, pca_dim=0, balancing_name='', file_appendix=''):
    kfold = StratifiedKFold(n_splits=N_SPLITS,
                            shuffle=True,
                            random_state=RANDOM_SEED)

    if pca_dim > 0:
        pca_name = 'PCA' + str(pca_dim) + '_'
        long_model_name = model_name + '_PCA' + str(pca_dim)
    else:
        pca_name = ''
        long_model_name = model_name
    models_list = load_models(long_model_name, balancing_name)
    evaluation_path = "Models/Evaluation/" + model_name + '_' + pca_name + balancing_name + file_appendix + ".csv"
    cm_norm_path = "Models/Evaluation/Confusion Matrix/" + model_name + '_' + pca_name + balancing_name + file_appendix + ".png"

    if os.path.exists(evaluation_path) or os.path.exists(cm_norm_path):
        print('Evaluation or confusion matrices already exists.')
    else:
        r_rmses, r_mses, r_maes = [], [], []
        accuracies, log_losses = [], []
        cms, cms_norm = [], []

        for i, (train_index, test_index) in enumerate(kfold.split(indep, dep)):
            print('Testing validation #' + str(i) + '...')
            model = models_list[i]

            # Get the validation data
            indep_test = indep[test_index, :]
            dep_test = dep[test_index]

            if pca_dim > 0:
                pca_path = 'Models/PCA/PCA' + "_val_" + str(i) + '.pkl'
                pca = pk.load(open(pca_path, 'rb'))
                indep_test = pca.transform(indep_test)[:, :pca_dim]

            predictions = model.predict(indep_test)

            r_rmses.append(root_mean_squared_error(dep_test, predictions))
            r_mses.append(mean_squared_error(dep_test, predictions))
            r_maes.append(mean_absolute_error(dep_test, predictions))

            accuracies.append(accuracy_score(dep_test, predictions))
            log_losses.append(log_loss(dep_test, model.predict_proba(indep_test)))

            cms.append(confusion_matrix(dep_test, predictions, normalize=None))
            cms_norm.append(confusion_matrix(dep_test, predictions, normalize='true'))

        r_rmses.append(fmean(r_rmses))
        r_mses.append(fmean(r_mses))
        r_maes.append(fmean(r_maes))
        accuracies.append(fmean(accuracies))
        log_losses.append(fmean(log_losses))

        cms_norm_average = np.mean(np.array(cms_norm), axis=0)
        disp_norm = ConfusionMatrixDisplay(cms_norm_average, display_labels=CLASS_NAMES)
        disp_norm.plot().figure_.savefig(cm_norm_path)
        cms_norm_stacked = np.vstack(tuple([cms_norm_average] + cms_norm))
        cms_index = [[str(i), '', '', '', ''] for i in range(N_SPLITS)]
        cms_index_flattened = ['AVG', '', '', '', ''] + [item for row in cms_index for item in row]
        data_cms_norm = pd.DataFrame(cms_norm_stacked, index=cms_index_flattened,
                                     columns=['Predicted 1', 'Predicted 2', 'Predicted 3', 'Predicted 4',
                                              'Predicted 5'])
        data_cms_norm.insert(loc=0, column='',
                             value=['True 1', 'True 2', 'True 3', 'True 4', 'True 5'] * (N_SPLITS + 1))

        cms_average = np.mean(np.array(cms), axis=0)
        cms_stacked = np.vstack(tuple([cms_average] + cms))
        data_cms = pd.DataFrame(cms_stacked, index=cms_index_flattened,
                                columns=['Predicted 1', 'Predicted 2', 'Predicted 3', 'Predicted 4', 'Predicted 5'])
        data_cms.insert(loc=0, column='', value=['True 1', 'True 2', 'True 3', 'True 4', 'True 5'] * (N_SPLITS + 1))

        data_dict = {'Rounded RMSE': r_rmses,
                     'Rounded MSE': r_mses,
                     'Rounded MAE': r_maes,
                     'Accuracy': accuracies,
                     'Log Loss': log_losses}

        data = pd.DataFrame(data_dict)
        data.rename(index={N_SPLITS: 'AVG'}, inplace=True)
        data.loc[len(data)] = pd.Series(dtype='float64')
        data.rename(index={N_SPLITS + 1: ''}, inplace=True)
        data.to_csv(evaluation_path)

        data_cms_norm.to_csv(evaluation_path, mode='a')
        data_cms.to_csv(evaluation_path, mode='a')


plt.ioff()
df = pd.read_parquet('Data/Vectorized/costco_2021_reviews_filtered_vectorized_final.parquet')

rating_counts = df['rating'].value_counts()
print('Number of ratings in training:')
print(np.rint(rating_counts * 8 / N_SPLITS))

X = np.array(df['vector'].tolist())
y = np.array(df['rating'].tolist())

# Train-test split
X_0, X_final_test, y_0, y_final_test = train_test_split(X, y,
                                                        shuffle=True,
                                                        random_state=RANDOM_SEED,
                                                        stratify=y,
                                                        test_size=0.2)


# # Scatter plot for PCA(n_components=2)
# X_reduced = PCA(n_components=2).fit_transform(X_0)
# colors = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'k'}
#
# fig = plt.figure(1, figsize=(8, 6))
#
# plt.scatter(
#     X_reduced[:, 0],
#     X_reduced[:, 1],
#     c=[colors[r] for r in y_0],
#     s=1,
# )

# Example usage
#
# save_models_kfold(X_0, y_0,
#                   model=LogisticRegression(multi_class='ovr', solver='liblinear'),
#                   model_appendix='ovr_liblinear',
#                   balancing=RandomUnderSampler(random_state=RANDOM_SEED),
#                   balancing_appendix='',
#                   balancing2=None,
#                   balancing2_appendix='')
#
# evaluate_classification(X_0, y_0, 'LogisticRegression_ovr_liblinear', 8, 'RandomUnderSampler')
