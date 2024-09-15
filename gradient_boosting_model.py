import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(train_file, test_file):
    """
    Load and preprocess training and test datasets.
    """
    # Load datasets
    train_df = pd.read_csv(train_file, encoding='unicode_escape').iloc[:500000]
    test_df = pd.read_csv(test_file, encoding='unicode_escape').iloc[:500000]
    # Drop unwanted columns
    columns_to_drop = [
        'Id', 'IncidentId', 'AlertId', 'Sha256', 'Url', 'ActionGrouped', 'ActionGranular', 'EmailClusterId', 'RegistryKey',
        'AntispamDirection', 'SuspicionLevel', 'LastVerdict', 'OSVersion', 'OSFamily', 'Roles', 'ResourceType',
        'ThreatFamily', 'OAuthApplicationId', 'ApplicationName', 'ApplicationId', 'RegistryValueData', 'RegistryValueName',
        'EmailClusterId', 'RegistryKey', 'NetworkMessageId', 'RegistryValueName', 'RegistryValueData', 'ApplicationId',
        'ApplicationName', 'OAuthApplicationId', 'ResourceType', 'ResourceIdName','MitreTechniques'
    ]
    
    train_df.drop(columns=columns_to_drop, axis=1, inplace=True)
    test_df.drop(columns=columns_to_drop + ['Usage'], axis=1, inplace=True)

    # Convert Timestamp to datetime and extract features
    for df in [train_df, test_df]:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        df['Day'] = df['Timestamp'].dt.day
        df['Hour'] = df['Timestamp'].dt.hour
        df['Minute'] = df['Timestamp'].dt.minute
        df['Second'] = df['Timestamp'].dt.second
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df.drop(['Timestamp'], axis=1, inplace=True)

    # Encode categorical features
    combined_df = pd.concat([train_df, test_df], axis=0)
    label_cols = ['Category', 'EntityType', 'EvidenceRole','City','State']
    for col in label_cols:
        model = LabelEncoder()
        combined_df[col] = model.fit_transform(combined_df[col])

    # Split the combined_df back to train_df and test_df
    train_df = combined_df.iloc[:len(train_df), :]
    test_df = combined_df.iloc[len(train_df):, :]

    # Ensure columns are hashable before dropping duplicates
    train_df = train_df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
    test_df = test_df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)

    # Drop duplicates and rows with NaNs
    train_df.drop_duplicates(inplace=True)
    test_df.drop_duplicates(inplace=True)
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    return train_df, test_df

def split_and_scale_data(train_df):
    """
    Split and scale the data, and apply SMOTE.
    """
    # Separate features and target variable
    X = train_df.drop(['IncidentGrade'], axis=1)
    y = train_df['IncidentGrade']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Apply SMOTE to handle imbalanced data
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    return x_train_resampled, y_train_resampled, x_test, y_test

def apply_pca(x_train_resampled, x_test):
    """
    Apply PCA to the scaled and resampled data.
    """
    pca = PCA(n_components=0.95)
    x_train_pca = pca.fit_transform(x_train_resampled)
    x_test_pca = pca.transform(x_test)

    return x_train_pca, x_test_pca

def evaluate_model(model, X_train, y_train, X_test, y_test, stage=""):
    """
    Evaluate the model using cross-validation.
    """
    scoring = {
        'accuracy': 'accuracy',
        'macro_f1': make_scorer(f1_score, average='macro'),
        'precision': 'precision_macro',
        'recall': 'recall_macro'
    }
    cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    
    print(f"{stage} Evaluation Metrics (Cross-validation):")
    for metric, scores in cv_results.items():
        if metric.startswith('test_'):
            print(f"{metric}: {scores.mean()}")
    
    # Training metrics
    y_train_pred = model.predict(X_train)
    print(f"\n{stage} Training Metrics:")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred)}")
    print(f"F1 Score: {f1_score(y_train, y_train_pred, average='macro')}")
    print(f"Precision: {precision_score(y_train, y_train_pred, average='macro')}")
    print(f"Recall: {recall_score(y_train, y_train_pred, average='macro')}")
    
    # Test metrics
    y_test_pred = model.predict(X_test)
    print(f"\n{stage} Test Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_test_pred, average='macro')}")
    print(f"Precision: {precision_score(y_test, y_test_pred, average='macro')}")
    print(f"Recall: {recall_score(y_test, y_test_pred, average='macro')}")
    print(f"Classification Report:\n{classification_report(y_test, y_test_pred)}")
    

def plot_feature_importance(model, X):
    """
    Plot feature importance if the model has feature_importances_ attribute.
    """
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title(f"Feature Importance for {model.__class__.__name__}")
        plt.show()

def main():
    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data("GUIDE_Train.csv", "GUIDE_Test.csv")

    # Split, scale, and apply SMOTE
    x_train_resampled, y_train_resampled, x_test, y_test = split_and_scale_data(train_df)

    # Apply PCA
    x_train_pca, x_test_pca = apply_pca(x_train_resampled, x_test)

    # Initialize GradientBoostingClassifier and evaluate before hyperparameter tuning
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(x_train_pca, y_train_resampled)
    print("Evaluation before hyperparameter tuning:")
    evaluate_model(gbc, x_train_pca, y_train_resampled, x_test_pca, y_test, stage="Train")

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train_pca, y_train_resampled)

    best_gbc_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print("Evaluation after hyperparameter tuning:")
    evaluate_model(best_gbc_model, x_train_pca, y_train_resampled, x_test_pca, y_test, stage="Train")

    # Plot feature importance
    plot_feature_importance(best_gbc_model, train_df.drop(['IncidentGrade'], axis=1))

if __name__ == "__main__":
    main()
