import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class KMeansClustering:
    def __init__(self,
                 n_clusters=4,
                 variance_threshold=0.01,
                 correlation_threshold=0.95,
                 mi_threshold=0.01):

        self.df = pd.read_csv('/Users/pintu/PycharmProjects/clustering/src/user_category/staging_202506271527.csv')
        self.identifiers = ['account_id', 'asset_id']
        self.features = ['count', 'avg_amount', 'std_amount', 'max_amount', 'wallet_age',
                         'frequency', 'unique_destination_count']
        self.fields = self.identifiers + self.features

        self.X = self.df[self.fields]
        self.X_train, self.X_test = train_test_split(self.X, test_size=0.2, random_state=42)

        self.scaled_X_train = []
        self.scaler = StandardScaler()

        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mi_threshold = mi_threshold

        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)

        self.inertia = 0

    def fit(self, data):
        scaled_data = self.scaler.fit_transform(data[self.features])
        scaled_df = pd.DataFrame(scaled_data, columns=self.features, index=data.index)
        self.kmeans.fit(scaled_df)

    def predict(self, data):
        scaled_data = self.scaler.transform(data[self.features])
        scaled_df = pd.DataFrame(scaled_data, columns=self.features, index=data.index)
        clusters = self.kmeans.predict(scaled_df)

        self.scaled_X_train = scaled_df
        self.inertia = self.kmeans.inertia_

        return clusters

    def plot_clusters(self, data):
        centers = pd.DataFrame(
            self.kmeans.cluster_centers_,
            columns=self.features)
        self.interpret_clusters(centers, self.scaled_X_train)
        print(centers)

        # inverted_centers = self.scaler.inverse_transform(centers)
        # print(inverted_centers)

        n_features = len(self.features) - 1
        height = 1.5
        aspect = 1.5
        size = height * n_features
        g = sns.pairplot(
            data,
            hue='cluster', palette='viridis',
            height=height, aspect=aspect,
        )
        g.fig.set_size_inches(size, size)
        g.fig.suptitle('User Category Clusters', fontsize=12)
        g.fig.subplots_adjust(top=0.95)
        # plt.tight_layout()
        plt.show(block=True)

    def interpret_clusters(self, centers, data):
        # compute percentiles for the whole clusters
        percentiles = {}
        for feature in self.features:
            percentiles[feature] = {
                "50": data[feature].quantile(0.5),
                "75": data[feature].quantile(0.75),
                "90": data[feature].quantile(0.9),
                "95": data[feature].quantile(0.95)
            }

        print(centers)
        for index, center in centers.iterrows():
            whale = 0
            robot = 0
            dormant = 0

            if center["count"] < percentiles["count"]["50"]:
                whale += 1
                dormant += 1
            elif center["count"] > percentiles["count"]["90"]:
                robot += 1

            if center["avg_amount"] < percentiles["avg_amount"]["50"]:
                robot += 1
            elif center["avg_amount"] > percentiles["avg_amount"]["95"]:
                whale += 1

            if center["frequency"] < percentiles["frequency"]["50"]:
                dormant += 1

            categories = {
                "whale": whale,
                "robot": robot,
                "dormant": dormant,
                "normal": 0.5,
            }
            max_key = max(categories, key=categories.get)
            centers.at[index, "label"] = max_key
            print(f"Key: {max_key}, Value: {categories[max_key]}")

    def filter_features_by_variance(self):
        variance = self.X_train[self.features].var()
        high_variance_features = variance[variance > self.variance_threshold].index.tolist()
        removed_features = variance[variance <= self.variance_threshold].index.tolist()

        print(f"selected features: {high_variance_features}")
        print(f"removed features: {removed_features}")

        self.features = high_variance_features
        self.fields = self.identifiers + self.features
        self.X_train = self.X_train[self.fields]
        self.X_test = self.X_test[self.fields]

    def filter_features_by_correlation(self):
        corr_matrix = self.X_train[self.features].corr().abs()

        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if
                              any(upper_tri[column] > self.correlation_threshold)]

        # Remove highly correlated features (keep the first one)
        features_to_remove = []
        for feature in high_corr_features:
            if feature not in features_to_remove:
                # Find all features highly correlated with this one
                correlated = corr_matrix[feature][corr_matrix[feature] > self.correlation_threshold].index.tolist()
                # Remove all except the first one
                features_to_remove.extend(correlated[1:])

        self.features = [f for f in self.features if f not in features_to_remove]
        self.fields = self.identifiers + self.features
        self.X_train = self.X_train[self.fields]
        self.X_test = self.X_test[self.fields]

        print(f"Correlation filtering - removed features: {features_to_remove}")
        print(f"Correlation filtering - remaining features: {self.features}")

    def filter_features_by_mi_score(self, n_clusters=3, top_k=None):
        # scale the data for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train[self.features])

        # perform clustering
        kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans_temp.fit_predict(X_scaled)

        # calculate mutual information between features and cluster labels
        mi_scores = mutual_info_classif(X_scaled, cluster_labels, random_state=42)

        # create results DataFrame
        mi_df = pd.DataFrame({
            'feature': self.features,
            'mi_with_clusters': mi_scores
        }).sort_values('mi_with_clusters', ascending=False)

        print(f"Mutual Information with Cluster Labels (k={n_clusters}):")
        print(mi_df.round(3))

        high_mi_features = mi_df.head(top_k)['feature'].tolist()
        removed_features = [f for f in self.features if f not in high_mi_features]

        print(f"Cluster MI filtering - selected features: {high_mi_features}")
        print(f"Cluster MI filtering - removed features: {removed_features}")

        self.features = high_mi_features
        self.fields = self.identifiers + self.features
        self.X_train = self.X_train[self.fields]
        self.X_test = self.X_test[self.fields]
