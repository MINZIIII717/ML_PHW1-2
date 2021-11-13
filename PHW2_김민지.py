import pandas as pd
import numpy as np
import itertools
import matplotlib as mpl

from scipy import linalg
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, estimate_bandwidth
from sklearn.cluster import MeanShift
from pyclustering.cluster.clarans import clarans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# dataset 읽어오기(csv file)
data_original = pd.read_csv('housing.csv')

# dataset 정보 확인 (nan value, type 등)
print(data_original.info())

# drop nan value
data_original.dropna(axis=0, inplace=True)
print(data_original.info())

data = data_original.drop(["median_house_value"], axis=1)


# Standard Scaler (input column)
def StandardScaling(col):
    from sklearn.preprocessing import StandardScaler
    standardScaler = StandardScaler()
    print(standardScaler.fit(col))
    train_data_standardScaled = standardScaler.fit_transform(col)
    print(train_data_standardScaled)
    return train_data_standardScaled


# MinMax Scaler (input column)
def MinMaxScaling(col):
    from sklearn.preprocessing import MinMaxScaler
    minMaxScaler = MinMaxScaler()
    train_data_minMaxScaled = minMaxScaler.fit_transform(col)
    return train_data_minMaxScaled


# MaxAbs Scaler (input column)
def MaxAbsScaling(col):
    from sklearn.preprocessing import MaxAbsScaler
    maxAbsScaler = MaxAbsScaler()
    print(maxAbsScaler.fit(col))
    train_data_maxAbsScaled = maxAbsScaler.fit_transform(col)
    return train_data_maxAbsScaled


# Robust Scaler (input column)
def RobustScaling(col):
    from sklearn.preprocessing import RobustScaler
    robustScaler = RobustScaler()
    print(robustScaler.fit(col))
    train_data_robustScaled = robustScaler.fit_transform(col)
    return train_data_robustScaled


# OneHot Encoder (input column)
def OneHotEncoding(col):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    encodedData = enc.fit_transform(col)
    encodedDataRecovery = np.argmax(encodedData, axis=1).reshape(-1, 1)
    return encodedDataRecovery


# Ordinal Encoder (input column)
def OrdinalEncoding(col):
    from sklearn.preprocessing import OrdinalEncoder
    ordinalEncoder = OrdinalEncoder()
    X = pd.DataFrame(col)
    ordinalEncoder.fit(X)
    return pd.DataFrame(ordinalEncoder.transform(X))


def kmeansClustering(dataset):
    list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    color = ['#1f77b4', '#ff7f0e', '#4c1318', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#d61328',
             '#7f7f7f', '#bcbd22', '#17becf', '#7f5f7f', '#bcbd82']
    best_score = -1
    retval = {}
    for i in list:
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(dataset)
        clusters = kmeans.predict(dataset)

        centers = pd.DataFrame(kmeans.cluster_centers_, columns=['p1', 'p2'])
        center_x = centers['p1']
        center_y = centers['p2']

        plt.scatter(dataset['pc 1'], dataset['pc 2'], c=kmeans.labels_, marker='o', s=10)
        plt.scatter(center_x, center_y, c=color[:1], marker="^", s=50)
        plt.show()

        score = silhouette_score(dataset, clusters)
        if score > best_score:
            retval = {"Algorithm": "KMeans", "best_cluster": i, "best_score": score}  # result dictionary
            best_score = score

    return retval


def dbscanClustering(dataset):
    list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    best_score = -1
    retval = {}
    for i in list:
        dbscan = DBSCAN(eps=i).fit(dataset)
        clusters = dbscan.fit_predict(dataset)
        core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        labels = dbscan.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 중복이므로 지울 라인

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)


            xy = dataset[class_member_mask & core_samples_mask]
            plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = dataset[class_member_mask & ~core_samples_mask]
            plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()


        if n_clusters_>1:
            score = silhouette_score(dataset, clusters)
            if score > best_score:
                retval = {"Algorithm": "DBSCAN", "best_eps": i, "best_score": score}  # result dictionary
                best_score = score
        else: retval = {"Algorithm": "DBSCAN", "best_eps": i, "best_score": -1}
    return retval


def plot_results(X, Y_, means, covariances, index, title):
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X.iloc[Y_ == i, 0], X.iloc[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


def emClustering(dataset):
    list = [2, 3, 5]
    best_score = -1
    retval = {}
    for i in list:
        em = GaussianMixture(n_components=i).fit(dataset)
        clusters = em.predict(dataset)
        plot_results(dataset, em.predict(dataset), em.means_, em.covariances_, 0, 'Gaussian Mixture')
        plt.show()

        score = silhouette_score(dataset, clusters)
        if score > best_score:
            retval = {"Algorithm": "EM", "best_component": i, "best_score": score}  # result dictionary
            best_score = score
    return retval


def meanshiftClustering(dataset):
    best_score = -1
    retval = {}
    # Compute clustering with MeanShift

    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(dataset, quantile=0.2, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(dataset)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    plt.figure(1)
    plt.clf()

    colors = itertools.cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(dataset.iloc[my_members, 0], dataset.iloc[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    clusters = ms.predict(dataset)
    retval = {"Algorithm": "MeanShift", "best_component": n_clusters_, "best_score": -1}  # result dictionary

    if n_clusters_ > 1:
        score = silhouette_score(dataset, clusters)
        if score > best_score:
            retval = {"Algorithm": "DBSCAN", "best_component": n_clusters_, "best_score": score}  # result dictionary
            best_score = score

    return retval


'''
def claransClustering(dataset):
    list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for i in list:
        cla = clarans(dataset, i, 6, 5)
        clusters = cla.get_clusters()
        ######################
        # scatter plot code
        #####################
'''


# cluster dataset and plotting
def clusteringModel(data_list):
    best_dataset = []
    best_result = {"Algorithm": "None", "best_cluster": 0, "best_score": -1}
    for dataset in data_list:

        temp = kmeansClustering(dataset)
        if best_result["best_score"] < temp["best_score"]:
            best_result = temp
            best_dataset = dataset.columns.values

        temp = dbscanClustering(dataset)
        if best_result["best_score"] < temp["best_score"]:
            best_result = temp
            best_dataset = dataset.columns.values

        temp = emClustering(dataset)
        if best_result["best_score"] < temp["best_score"]:
            best_result = temp
            best_dataset = dataset.columns.values

        temp = meanshiftClustering(dataset)
        if best_result["best_score"] < temp["best_score"]:
            best_result = temp
            best_dataset = dataset.columns.values

    return best_dataset, best_result


def main():
    # feature 목록 ("median_house_value" 제외)
    feature = ['longitude', "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
               "households",
               "median_income", 'ocean_proximity']
    # feature subset combination
    feature_combination = list(map(list, itertools.combinations(feature, 8)))

    # preprocessing된 feature combination subset list
    prepro_feature = []

    # 1:encoding 먼저 하고 scaling 진행 2:Scaling 먼저하고, Encoding 진행
    way = int(input('1:First Encoding, Second Scaling\n2:First Scaling, Second Scaling\n=>'))

    # 진행할 Encoding method 입력
    enc = input('Encoding Method:')

    # 진행할 Encoding method 입력
    scal = input('Scaling Method:')

    for f in feature_combination:
        print('-' * 60)
        print('Feature: {}\n'.format(f))

        temp = data[f].copy()

        for l in f:
            # 지정한 feature에 해당하는 column
            # col = (temp[l].values).reshape(-1, 1)
            # Encoding 먼저
            if way == 1:
                if enc == 'onehot':
                    # Encoding 진행한 결과 값을 encode_col에 저장
                    encode_col = OneHotEncoding(temp[l])

                elif enc == 'ordinal':
                    # Encoding 진행한 결과 값을 encode_col에 저장
                    encode_col = OrdinalEncoding(temp[l])

                if scal == 'standard':
                    # Scaling 진행한 결과값을 feature에 저장
                    temp[l] = StandardScaling(encode_col)

                elif scal == 'minmax':
                    # Scaling 진행한 결과값을 feature에 저장
                    temp[l] = MinMaxScaling(encode_col)

                elif scal == 'maxabs':
                    # Scaling 진행한 결과값을 feature에 저장
                    temp[l] = MaxAbsScaling(encode_col)

                elif scal == 'robust':
                    # Scaling 진행한 결과값을 feature에 저장
                    temp[l] = RobustScaling(encode_col)

            # Scaling 먼저 진행할 때
            elif way == 2:
                # 진행할 Scaling method 입력
                scal = input('Scaling Method:')
                if scal == 'standard':
                    # Scaling 진행한 결과값을 scaled_col에 저장
                    scaled_col = StandardScaling(temp[l])

                elif scal == 'minmax':
                    # Scaling 진행한 결과값을 scaled_col에 저장
                    scaled_col = MinMaxScaling(temp[l])

                elif scal == 'maxabs':
                    # Scaling 진행한 결과값을 scaled_col에 저장
                    scaled_col = MaxAbsScaling(temp[l])

                elif scal == 'robust':
                    # Scaling 진행한 결과값을 scaled_col에 저장
                    scaled_col = RobustScaling(temp[l])

                # 진행할 Encoding method 입력
                enc = input('Encoding Method:')
                if enc == 'onehot':
                    # Encoding 진행한 결과 값을 encode_col에 저장
                    temp[l] = OneHotEncoding(scaled_col)

                elif enc == 'ordinal':
                    # Encoding 진행한 결과 값을 encode_col에 저장
                    temp[l] = OrdinalEncoding(scaled_col)

        # ---------------- PCA를 이용해 고차원 data를 3차원으로 변환할 것 -----------------
        temp = temp.loc[:, f].values
        pca = PCA(n_components=2)
        principleComponents = pca.fit_transform(temp)
        principleDf = pd.DataFrame(data=principleComponents, columns=['pc 1', 'pc 2'])
        # ---------------------------------------------------------------------------

        prepro_feature.append(principleDf)

    print(prepro_feature)
    best_dataset, best_result = clusteringModel(prepro_feature)
    print(best_dataset)
    print(best_result)


if __name__ == "__main__":
    main()
