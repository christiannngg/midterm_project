import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


def load_data(file_path: str, nrows: int = 150000) -> pd.DataFrame:
    """
    loads the dataset from a CSV file.

    Args:
        file_path (str): path to the CSV file.
        nrows (int, optional): number of rows to read. Defaults to 150000.

    Returns:
        pd.DataFrame: loaded dataset.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    df = pd.read_csv(file_path, nrows=nrows)
    print("\n" + "-" * 50)
    print(f" Dataset Loaded Successfully: {file_path}")
    print("-" * 50)
    print(f" Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print(f" Missing Values: {df.isnull().sum().sum()}")
    print("-" * 50 + "\n")

    return df


def preprocess_data(df: pd.DataFrame):
    """
    prepares the dataset by separating features and target,
    performing a train-test split, and normalizing features.

    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        tuple: scaled training and test sets, target values, and feature names.
    """
    # drop 'id' (if exists) and 'Class' (target variable)
    X = df.drop(columns=['id', 'Class'], errors='ignore')
    y = df['Class']

    # split dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # normalize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_distribution = pd.Series(y_train).value_counts(normalize=True) * 100
    print("\n" + "-" * 50)
    print("Class Distribution in Training Set:")
    print("-" * 50)
    print(f"Legitimate Transactions: {class_distribution[0]:.2f}%")
    print(f"Fraudulent Transactions: {class_distribution[1]:.2f}%")
    print("-" * 50 + "\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def train_model(X_train: np.ndarray, y_train: pd.Series) -> RandomForestClassifier:
    """
    trains a Random Forest classifier with cross-validation.

    Args:
        X_train (np.ndarray): scaled training feature set.
        y_train (pd.Series): training target values.

    Returns:
        RandomForestClassifier: trained model.
    """
    rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, min_samples_split=5, random_state=42)

    # perform 5-fold cross-validation and calculate F1-score
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')

    print("\n" + "-" * 50)
    print("Model Training Complete")
    print("-" * 50)
    print("Cross-validation F1 scores: ", ["{:.3f}".format(score) for score in cv_scores])
    print(f"Average F1 score: {np.mean(cv_scores):.3f}")
    print("-" * 50 + "\n")

    # train the model on the entire training set
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: pd.Series):
    """
    evaluates the trained model using classification metrics and a confusion matrix.

    Args:
        model (RandomForestClassifier): the trained model.
        X_test (np.ndarray): scaled test feature set.
        y_test (pd.Series): test target values.
    """
    y_pred = model.predict(X_test)

    print("\n" + "-" * 50)
    print("Model Performance on Test Set")
    print("-" * 50)
    print(classification_report(y_test, y_pred, digits=3))
    print("-" * 50 + "\n")

    # generate confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_feature_importance(model: RandomForestClassifier, feature_names: pd.Index):
    """
    plots the feature importance ranking.

    Args:
        model (RandomForestClassifier): the trained model.
        feature_names (pd.Index): feature names.
    """
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance',                                                                            ascending=False)

    print("\n" + "-" * 50)
    print("Top 5 Most Important Features:")
    print("-" * 50)
    print(feature_imp.head().to_string(index=False))
    print("-" * 50 + "\n")

    # generate feature importance bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_imp, x='Importance', y='Feature')
    plt.title('Feature Importance Ranking')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(X: pd.DataFrame):
    """
    plots the feature correlation matrix.

    Args:
        X (pd.DataFrame): feature dataset.
    """
    plt.figure(figsize=(12, 8))
    correlation_matrix = X.corr()

    # generate correlation heatmap
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model: RandomForestClassifier, X_test: np.ndarray, y_test: pd.Series):
    """
    plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        model (RandomForestClassifier): the trained model.
        X_test (np.ndarray): scaled test feature set.
        y_test (pd.Series): test target values.
    """
    # get predicted probabilities for the positive class
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # generate ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def main():
    """
    main function to execute the credit card fraud detection pipeline:
    1. load data
    2. preprocess data
    3. train model
    4. evaluate model
    5. visualize results
    """
    file_path = 'creditcard_2023.csv'  # CSV file location
    df = load_data(file_path)

    if df is None:
        return  # exit if the dataset is not found

    # preprocesses data and obtain scaled feature sets
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    # train the random forest model
    model = train_model(X_train, y_train)

    # evaluate model performance
    evaluate_model(model, X_test, y_test)

    # plot feature importance ranking
    plot_feature_importance(model, feature_names)

    # plot feature correlation matrix
    plot_correlation_matrix(df.drop(columns=['id', 'Class'], errors='ignore'))

    # plot ROC curve
    plot_roc_curve(model, X_test, y_test)


if __name__ == "__main__":
    main()  # run the script when executed

