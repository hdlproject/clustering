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

    def calculate_mutual_information_with_clusters(self, n_clusters=None):
        """
        Calculate mutual information between features and cluster labels.
        This is the most appropriate method for clustering applications.
        
        Steps:
        1. Perform clustering on the data
        2. Use cluster labels as 'target' variables
        3. Calculate MI between features and cluster labels
        """
        if n_clusters is None:
            n_clusters = self.n_clusters

        # Scale the data for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train[self.features])

        # Perform clustering
        kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans_temp.fit_predict(X_scaled)

        # Calculate mutual information between features and cluster labels
        mi_scores = mutual_info_classif(X_scaled, cluster_labels, random_state=42)

        # Create results DataFrame
        mi_df = pd.DataFrame({
            'feature': self.features,
            'mi_with_clusters': mi_scores
        }).sort_values('mi_with_clusters', ascending=False)

        print(f"Mutual Information with Cluster Labels (k={n_clusters}):")
        print(mi_df.round(3))

        return mi_df, cluster_labels

    def filter_features_by_cluster_mi(self, n_clusters=None, top_k=None, threshold=None):
        """
        Filter features based on mutual information with cluster labels.
        This is the recommended approach for clustering applications.
        """
        mi_df, cluster_labels = self.calculate_mutual_information_with_clusters(n_clusters)

        if threshold is not None:
            # Keep features above threshold
            high_mi_features = mi_df[mi_df['mi_with_clusters'] > threshold]['feature'].tolist()
        elif top_k is not None:
            # Keep top k features
            high_mi_features = mi_df.head(top_k)['feature'].tolist()
        else:
            # Use default threshold
            high_mi_features = mi_df[mi_df['mi_with_clusters'] > self.mi_threshold]['feature'].tolist()

        removed_features = [f for f in self.features if f not in high_mi_features]

        print(f"Cluster MI filtering - selected features: {high_mi_features}")
        print(f"Cluster MI filtering - removed features: {removed_features}")

        self.features = high_mi_features
        self.fields = self.identifiers + self.features
        self.X_train = self.X_train[self.fields]
        self.X_test = self.X_test[self.fields]

        return mi_df, cluster_labels

    def comprehensive_feature_selection(self):
        """
        Apply all feature selection methods in sequence.
        """
        print("=== Starting Comprehensive Feature Selection ===")
        print(f"Initial features: {self.features}")

        # Step 1: Variance filtering
        print("\n1. Variance filtering...")
        self.filter_features_by_variance()

        # Step 2: Correlation filtering
        print("\n2. Correlation filtering...")
        self.filter_features_by_correlation()

        # Step 3: Mutual Information with cluster labels (recommended for clustering)
        print("\n3. Mutual Information with cluster labels...")
        self.filter_features_by_cluster_mi(top_k=3)  # Keep top 3 features

        print(f"\nFinal selected features: {self.features}")
        print(f"Final dataset shape: {self.X_train.shape}")

    def calculate_mutual_information(self, target_feature=None):
        """
        Calculate mutual information between features.
        For clustering, we can use one feature as a 'target' to measure MI with others.
        """
        if target_feature is None:
            # Use the first feature as target for demonstration
            target_feature = self.features[0]

        # Prepare data: use one feature as target, others as features
        feature_cols = [f for f in self.features if f != target_feature]
        X = self.X_train[feature_cols]
        y = self.X_train[target_feature]

        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)

        # Create a DataFrame with feature names and MI scores
        mi_df = pd.DataFrame({
            'feature': feature_cols,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)

        print(f"Mutual Information scores (using {target_feature} as target):")
        print(mi_df)

        return mi_df

    def filter_features_by_mutual_information(self, target_feature=None, top_k=None):
        """
        Filter features based on mutual information scores.
        Keep top_k features with highest MI scores.
        """
        mi_df = self.calculate_mutual_information(target_feature)

        if top_k is None:
            # Keep features above threshold
            high_mi_features = mi_df[mi_df['mutual_info'] > self.mi_threshold]['feature'].tolist()
        else:
            # Keep top k features
            high_mi_features = mi_df.head(top_k)['feature'].tolist()

        # Always include the target feature if it was used
        if target_feature and target_feature not in high_mi_features:
            high_mi_features.append(target_feature)

        removed_features = [f for f in self.features if f not in high_mi_features]

        print(f"MI filtering - selected features: {high_mi_features}")
        print(f"MI filtering - removed features: {removed_features}")

        self.features = high_mi_features
        self.fields = self.identifiers + self.features
        self.X_train = self.X_train[self.fields]
        self.X_test = self.X_test[self.fields]
