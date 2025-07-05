import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class KMeansClustering:
    def __init__(self, n_clusters=4):
        self.df = pd.read_csv('/Users/pintu/PycharmProjects/clustering/src/user_category/staging_202506271527.csv')
        self.features = ['account_id', 'asset_id', 'count', 'avg_amount', 'std_amount', 'max_amount', 'wallet_age',
                         'frequency', 'unique_destination_count']
        self.features_to_scale = ['count', 'avg_amount', 'std_amount', 'max_amount', 'wallet_age',
                                  'frequency', 'unique_destination_count']
        self.X = self.df[self.features]
        self.X_train, self.X_test = train_test_split(self.X, test_size=0.2, random_state=42)
        self.scaled_X_train = []
        self.inertia = 0

        self.scaler = StandardScaler()

        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)

    def fit(self, data):
        scaled_data = self.scaler.fit_transform(data[self.features_to_scale])
        scaled_df = pd.DataFrame(scaled_data, columns=self.features_to_scale, index=data.index)
        self.kmeans.fit(scaled_df)

    def predict(self, data):
        scaled_data = self.scaler.transform(data[self.features_to_scale])
        scaled_df = pd.DataFrame(scaled_data, columns=self.features_to_scale, index=data.index)
        clusters = self.kmeans.predict(scaled_df)

        self.scaled_X_train = scaled_df
        self.inertia = self.kmeans.inertia_

        return clusters

    def plot_clusters(self, data):
        centers = pd.DataFrame(
            self.kmeans.cluster_centers_,
            columns=self.features_to_scale)
        self.interpret_clusters(centers, self.scaled_X_train)
        print(centers)

        # inverted_centers = self.scaler.inverse_transform(centers)
        # print(inverted_centers)

        n_features = len(self.features_to_scale) - 1
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
        for feature in self.features_to_scale:
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
