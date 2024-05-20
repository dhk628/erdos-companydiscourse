import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

df = pd.read_parquet('Data/Vectorized/costco_2021_reviews_filtered_vectorized_master_en_float32_nodup.parquet')

rating_counts = df['rating'].value_counts()
print('Number of ratings:')
print(rating_counts)

data5 = pd.DataFrame(df[df['rating'] == 5]).iloc[:rating_counts.min()]
data4 = pd.DataFrame(df[df['rating'] == 4]).iloc[:rating_counts.min()]
data3 = pd.DataFrame(df[df['rating'] == 3]).iloc[:rating_counts.min()]
data2 = pd.DataFrame(df[df['rating'] == 2]).iloc[:rating_counts.min()]
data1 = pd.DataFrame(df[df['rating'] == 1]).iloc[:rating_counts.min()]

data5_train, data5_test = train_test_split(data5, random_state=1, test_size=.2)
data4_train, data4_test = train_test_split(data4, random_state=1, test_size=.2)
data3_train, data3_test = train_test_split(data3, random_state=1, test_size=.2)
data2_train, data2_test = train_test_split(data2, random_state=1, test_size=.2)
data1_train, data1_test = train_test_split(data1, random_state=1, test_size=.2)

data_train = pd.concat([data5_train, data4_train, data3_train, data2_train, data1_train])
data_test = pd.concat([data5_test, data4_test, data3_test, data2_test, data1_test])

vectors_train = np.array(data_train['vector'].tolist())
vectors_test = np.array(data_test['vector'].tolist())

ratings_train = np.array(data_train['rating'].tolist())
ratings_test = np.array(data_test['rating'].tolist())


# Logistic Regression, One-vs-All
logreg = LogisticRegression(multi_class='ovr', solver='liblinear')
logreg.fit(vectors_train, ratings_train)

ratings_pred_logreg = logreg.predict(vectors_test)
cm_logreg = confusion_matrix(ratings_test, ratings_pred_logreg)
print('Confusion matrix for one-vs-all logistic regression:')
print(cm_logreg)


# Logistic Regression, Multinomial
logreg_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
logreg_multi.fit(vectors_train, ratings_train)

ratings_pred_logreg_multi = logreg_multi.predict(vectors_test)
cm_logreg_multi = confusion_matrix(ratings_test, ratings_pred_logreg_multi)
print('Confusion matrix for multinomial logistic regression:')
print(cm_logreg_multi)


# Decision Tree
depth = 10
dtree = DecisionTreeClassifier(max_depth=depth).fit(vectors_train, ratings_train)
ratings_pred_dtree = dtree.predict(vectors_test)

# creating a confusion matrix
cm_dtree = confusion_matrix(ratings_test, ratings_pred_dtree)
print('Confusion matrix for decision tree (depth ' + str(depth) + '):')
print(cm_dtree)


# SVM
svm_linear = SVC(kernel='linear', C=1).fit(vectors_train, ratings_train)
ratings_pred_svm = svm_linear.predict(vectors_test)


# # Plotting
# class_names = {1,2,3,4,5}
# disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_logreg_multi, display_labels=class_names)
# disp.plot()
# plt.show()


# # PCA
# X_reduced = PCA(n_components=2).fit_transform(vectors_train)
# colors = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'k'}
#
# fig = plt.figure(1, figsize=(8, 6))
#
# plt.scatter(
#     X_reduced[:, 0],
#     X_reduced[:, 1],
#     c=[colors[r] for r in ratings_train],
#     s=1,
# )
#
# # n = len(data1_train)
# # plt.scatter(
# #     X_reduced[:n, 0],
# #     X_reduced[:n, 1],
# #     c='k',
# #     s=1,
# # )
# #
# # plt.scatter(
# #     X_reduced[4*n:5*n, 0],
# #     X_reduced[4*n:5*n, 1],
# #     c='r',
# #     s=1,
# # )
#
# plt.show()


