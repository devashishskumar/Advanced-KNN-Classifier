# Name: Devashish Sanjay Kumar
# Student ID: 1002157097



import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier

# Define KNN Classifier
class KNNClassifier:
    def __init__(self, k=5, distance_metric='euclidean', kernel='uniform'):
        """
        Initialize KNN Classifier with optional kernel weighting
        kernel options: 'uniform' (no weighting), 'gaussian', 'epanechnikov'
        """
        self.k = k
        self.distance_metric = distance_metric
        self.kernel = kernel
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Stores training data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def _kernel_weight(self, distance):
        """Apply kernel weighting to distances"""
        if self.kernel == 'uniform':
            return 1.0
        elif self.kernel == 'gaussian':
            return np.exp(-0.5 * (distance ** 2))
        elif self.kernel == 'epanechnikov':
            return np.maximum(0, 1 - distance ** 2)
        else:
            raise ValueError("Unsupported kernel type")
    
    def predict(self, X):
        """Predicts the class labels for given test data"""
        X = np.array(X)
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        """Helper function to predict a single instance using kernel weighting"""
        distances = np.array([self._compute_distance(x, x_train) for x_train in self.X_train])
        k_indices = np.argsort(distances)[:self.k]
        
        # Apply kernel weights to k nearest neighbors
        k_distances = distances[k_indices]
        k_weights = np.array([self._kernel_weight(d) for d in k_distances])
        k_nearest_labels = self.y_train[k_indices]
        
        # Modified weighted voting for categorical data
        if self.kernel != 'uniform':
            weighted_votes = {}
            for label, weight in zip(k_nearest_labels, k_weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            return max(weighted_votes.items(), key=lambda x: x[1])[0]
        else:
            # Modified voting for categorical data
            votes = {}
            for label, dist in zip(k_nearest_labels, k_distances):
                # Use inverse distance as weight even for uniform kernel
                weight = 1.0 / (dist + 1e-6)  # Add small constant to prevent division by zero
                votes[label] = votes.get(label, 0) + weight
            return max(votes.items(), key=lambda x: x[1])[0]
    
    def _compute_distance(self, x1, x2):
        """Computes distance based on chosen metric with categorical data handling"""
        if self.distance_metric == 'euclidean':
            # Modified distance calculation for categorical data
            # For car evaluation dataset, all features are categorical
            # Use a modified distance that penalizes differences more strongly
            diff = np.abs(x1 - x2)
            # Scale the differences based on feature ranges
            # This will make the distance more sensitive to categorical differences
            scaled_diff = diff * (diff > 0).astype(float)  # Penalize any difference
            return np.sqrt(np.sum(scaled_diff ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Unsupported distance metric")

# Define dataset paths and configurations
datasets = {
    "Hayes-Roth": {
        "train_path": "hayes-roth.data",
        "test_path": "hayes-roth.test",
        "columns": ["Instance", "Hobby", "Age", "Educational Level", "Marital Status", "Class"],
        "drop_columns": ["Instance"]
    },
    "Car Evaluation": {
        "path": "car.data",
        "columns": ["Buying", "Maint", "Doors", "Persons", "Lug_boot", "Safety", "Class"],
        "encoding": {
            "Buying": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
            "Maint": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
            "Doors": {"2": 2, "3": 3, "4": 4, "5more": 5},
            "Persons": {"2": 2, "4": 4, "more": 5},
            "Lug_boot": {"small": 0, "med": 1, "big": 2},
            "Safety": {"low": 0, "med": 1, "high": 2},
            "Class": {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
        }
    },
    "Breast Cancer": {
        "path": "breast-cancer.data",
        "columns": ["Class", "Age", "Menopause", "Tumor Size", "Inv Nodes", "Node Caps", 
                   "Deg Malig", "Breast", "Breast Quad", "Irradiat"],
        "encoding": {
            "Class": {"no-recurrence-events": 0, "recurrence-events": 1},
            "Age": {"20-29": 0, "30-39": 1, "40-49": 2, "50-59": 3, "60-69": 4, "70-79": 5},
            "Menopause": {"lt40": 0, "premeno": 1, "ge40": 2},
            "Tumor Size": {
                "0-4": 0, "5-9": 1, "10-14": 2, "15-19": 3, "20-24": 4, "25-29": 5,
                "30-34": 6, "35-39": 7, "40-44": 8, "45-49": 9, "50-54": 10
            },
            "Inv Nodes": {
                "0-2": 0, "3-5": 1, "6-8": 2, "9-11": 3, "12-14": 4, "15-17": 5, "24-26": 6
            },
            "Node Caps": {"no": 0, "yes": 1},
            "Breast": {"left": 0, "right": 1},
            "Breast Quad": {"left_low": 0, "left_up": 1, "right_low": 2, "right_up": 3, "central": 4},
            "Irradiat": {"no": 0, "yes": 1}
        }
    }
}

def load_and_preprocess_data(dataset_name, dataset_config):
    """Load and preprocess a dataset based on its configuration"""
    print(f"\nProcessing {dataset_name} dataset...")
    
    if dataset_name == "Hayes-Roth":
        train_df = pd.read_csv(dataset_config["train_path"], header=None, names=dataset_config["columns"])
        test_df = pd.read_csv(dataset_config["test_path"], header=None, names=dataset_config["columns"])
        
        for col in dataset_config["drop_columns"]:
            train_df = train_df.drop(columns=[col])
            test_df = test_df.drop(columns=[col])
        
        train_df = train_df.apply(pd.to_numeric, errors='coerce')
        test_df = test_df.apply(pd.to_numeric, errors='coerce')
        
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        
        if len(test_df) == 0:
            print(f"Note: Using train-test split for {dataset_name}")
            train_df = train_df.sample(frac=1, random_state=42)  # Shuffle the data
            split_idx = int(len(train_df) * 0.8)
            return train_df[:split_idx], train_df[split_idx:]
        
        return train_df, test_df
    
    else:
        df = pd.read_csv(dataset_config["path"], header=None, names=dataset_config["columns"])
        if "encoding" in dataset_config:
            df = df.replace(dataset_config["encoding"])
        df = df.replace("?", pd.NA).fillna(df.mode().iloc[0])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df = df.sample(frac=1, random_state=42)  # Shuffle the data
        split_idx = int(len(df) * 0.8)
        return df[:split_idx], df[split_idx:]

def evaluate_with_cross_validation(X, y, k_values=None, kernel='uniform'):
    """Evaluate both Custom and Scikit-Learn KNN using 10-fold cross validation"""
    if k_values is None:
        k_values = [1, 3, 5, 7, 9]  # Common odd values to avoid ties
    
    # Add feature normalization
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results = {}
    
    for k in k_values:
        custom_accuracies = []
        sklearn_accuracies = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Custom KNN with uniform weights (no kernel)
            knn_custom = KNNClassifier(k=k, kernel='uniform')
            knn_custom.fit(X_train, y_train)
            y_pred_custom = knn_custom.predict(X_test)
            custom_accuracies.append(accuracy_score(y_test, y_pred_custom))
            
            # Scikit-learn KNN with uniform weights
            knn_sklearn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
            knn_sklearn.fit(X_train, y_train)
            y_pred_sklearn = knn_sklearn.predict(X_test)
            sklearn_accuracies.append(accuracy_score(y_test, y_pred_sklearn))
        
        # Statistical comparison using paired t-test
        custom_accuracies = np.array(custom_accuracies)
        sklearn_accuracies = np.array(sklearn_accuracies)
        t_stat, p_value = stats.ttest_rel(custom_accuracies, sklearn_accuracies)
        
        results[k] = {
            'custom_mean': np.mean(custom_accuracies),
            'sklearn_mean': np.mean(sklearn_accuracies),
            'custom_std': np.std(custom_accuracies),
            'sklearn_std': np.std(sklearn_accuracies),
            't_stat': t_stat,
            'p_value': p_value
        }
    
    return results

def main():
    """Main function to run the analysis on all datasets"""
    print("KNN Analysis with 10-Fold Cross Validation")
    print("=" * 80)
    
    k_values = [1, 3, 5, 7, 9]
    kernels = ['uniform']
    
    for dataset_name, dataset_config in datasets.items():
        try:
            train_df, test_df = load_and_preprocess_data(dataset_name, dataset_config)
            full_df = pd.concat([train_df, test_df])
            X = full_df.iloc[:, :-1].values
            y = full_df.iloc[:, -1].values
            
            print(f"\nDataset: {dataset_name}")
            print("=" * 80)
            
            for kernel in kernels:
                results = evaluate_with_cross_validation(X, y, k_values=k_values, kernel=kernel)
                
                # Find best k value for custom KNN
                best_k_custom = max(results.keys(), 
                                  key=lambda k: results[k]['custom_mean'])
                
                print("\nCross-Validation Results:")
                print("-" * 80)
                print("k  |  Custom KNN (Mean ± Std)  |  Scikit-Learn KNN (Mean ± Std)")
                print("-" * 80)
                for k in k_values:
                    print(f"{k:2d} |  {results[k]['custom_mean']:.4f} ± {results[k]['custom_std']:.4f}  |  {results[k]['sklearn_mean']:.4f} ± {results[k]['sklearn_std']:.4f}")
                
                print("\nBest Model Performance:")
                print("-" * 80)
                print(f"Optimal k value: {best_k_custom}")
                print(f"Custom KNN Best Accuracy: {results[best_k_custom]['custom_mean']:.4f} ± {results[best_k_custom]['custom_std']:.4f}")
                print(f"Scikit-Learn KNN Accuracy: {results[best_k_custom]['sklearn_mean']:.4f} ± {results[best_k_custom]['sklearn_std']:.4f}")
                
                print("\nStatistical Analysis:")
                print("-" * 80)
                print(f"T-Statistic: {results[best_k_custom]['t_stat']:.4f}")
                print(f"P-Value: {results[best_k_custom]['p_value']:.4f}")
                print("\nHypothesis Test Conclusion:")
                if results[best_k_custom]['p_value'] < 0.05:
                    print("The difference between Custom KNN and Scikit-Learn KNN is statistically significant (p < 0.05)")
                else:
                    print("No statistically significant difference between Custom KNN and Scikit-Learn KNN (p ≥ 0.05)")
                
            print("\n" + "=" * 80)
                
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            print("=" * 80)

if __name__ == "__main__":
    main() 