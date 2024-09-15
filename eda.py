
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
        'ApplicationName', 'OAuthApplicationId', 'ResourceType', 'ResourceIdName'
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

    # Handle MitreTechniques column
    mlb = MultiLabelBinarizer()
    for df in [train_df, test_df]:
        df['MitreTechniques'] = df['MitreTechniques'].fillna('').apply(lambda x: x.split(';'))
        mitre_encoded = mlb.fit_transform(df['MitreTechniques'])
        mitre_techniques = pd.DataFrame(mitre_encoded, columns=mlb.classes_, index=df.index)
        df.drop(['MitreTechniques'], axis=1, inplace=True)
        df = pd.concat([df, mitre_techniques], axis=1)

    # Encode categorical features
    combined_df = pd.concat([train_df, test_df], axis=0)
    label_cols = ['Category', 'EntityType', 'EvidenceRole']
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

    # Apply PCA
    pca = PCA(n_components=0.95)
    x_train_pca = pca.fit_transform(x_train_resampled)
    x_test_pca = pca.transform(x_test)

    return x_train_pca, x_test_pca, y_train_resampled, y_test

def evaluate_model(model, X, y):
    """Evaluate the model using cross-validation."""
    scoring = {
        'accuracy': 'accuracy',
        'macro_f1': make_scorer(f1_score, average='macro'),
        'precision': 'precision_macro',
        'recall': 'recall_macro'
    }
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
    for metric, scores in cv_results.items():
        if metric.startswith('test_'):
            print(f"{metric}: {scores.mean()}")

def plot_feature_importance(model, X):
    """Plot feature importance for models with feature_importances_ attribute."""
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
    """Main function to run the data processing, model training, and evaluation."""
    train_df, test_df = load_and_preprocess_data("GUIDE_Train.csv", "GUIDE_Test.csv")
    
    # Univariate Analysis
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Category', data=train_df)
    plt.title('Category Distribution')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['IncidentGrade'], bins=30, kde=True)
    plt.title('Incident Grade Distribution')
    plt.show()

    # Bivariate Analysis
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='AccountName', y='IncidentGrade', data=train_df)
    plt.title('Incident Grade Over Time')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='IncidentGrade', y='Category', data=train_df)
    plt.title('Incident Grade by Category')
    plt.show()

    # OrgId Analysis and Plot
    train_df['OrgId'] = (train_df['OrgId'] - train_df['OrgId'].mean()) / train_df['OrgId'].std()
    train_df = train_df[(train_df['OrgId'] > -2) & (train_df['OrgId'] < 2)]
    sns.boxplot(train_df['OrgId'])
    plt.show()
    train_df['OrgId'] = (train_df['OrgId'] - train_df['OrgId'].min()) / (train_df['OrgId'].max() - train_df['OrgId'].min())

    # DetectorId Analysis and Plot
    train_df['DetectorId'] = (train_df['DetectorId'] - train_df['DetectorId'].mean()) / train_df['DetectorId'].std()
    train_df = train_df[(train_df['DetectorId'] > -2) & (train_df['DetectorId'] < 2)]
    sns.boxplot(train_df['DetectorId'])
    plt.show() 
    train_df['DetectorId'] = (train_df['DetectorId'] - train_df['DetectorId'].min()) / (train_df['DetectorId'].max() - train_df['DetectorId'].min())

    x_train_pca, x_test_pca, y_train_resampled, y_test = split_and_scale_data(train_df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    for model_name, model in models.items():
        print(f"Evaluating {model_name}")
        model.fit(x_train_pca, y_train_resampled)
        evaluate_model(model, x_train_pca, y_train_resampled)
        
        y_pred = model.predict(x_test_pca)
        print(f"Testing Metrics for {model_name}:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro')}")
        print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
        print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Feature Importance Plot
        plot_feature_importance(model, pd.DataFrame(x_train_pca))

    # Hyperparameter Tuning with GridSearchCV
    param_grids = {
        "Logistic Regression": {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear']
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "Gradient Boosting": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }

    best_estimators = {}

    for model_name, param_grid in param_grids.items():
        print(f"Hyperparameter Tuning for {model_name}")
        grid_search = GridSearchCV(models[model_name], param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train_pca, y_train_resampled)
        best_estimators[model_name] = grid_search.best_estimator_
        print(f"Best Parameters for {model_name}: {grid_search.best_params_}")

    for model_name, best_estimator in best_estimators.items():
        print(f"Evaluating Tuned {model_name}")
        evaluate_model(best_estimator, x_train_pca, y_train_resampled)
        
        y_pred = best_estimator.predict(x_test_pca)
        print(f"Tuned Testing Metrics for {model_name}:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro')}")
        print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
        print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
        # Feature Importance Plot
        plot_feature_importance(best_estimator, pd.DataFrame(x_train_pca))

if __name__ == "__main__":
    main()
