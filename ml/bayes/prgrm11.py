import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def evaluate_gnb(file_path, target_col, dataset_name, label_encode=True, drop_cols=None):
    # Load data
    data = pd.read_csv(file_path)
    
    # Drop unwanted columns if any
    if drop_cols:
        data = data.drop(drop_cols, axis=1, errors='ignore')
    
    # Separate features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Encode target labels if needed
    if label_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    print(f"\n==== {dataset_name} Dataset ====")
    
    for test_ratio in [0.2, 0.3, 0.4]:
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        
        # Train model
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        print(f"\nTrain-Test Split: {int((1-test_ratio)*100)}-{int(test_ratio*100)}")
        print(f"Accuracy: {round(accuracy, 4)}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(cr)


# Run for all datasets

# Iris
evaluate_gnb('Iris.csv', target_col='Species', dataset_name='Iris')

# Diabetes (no label encoding)
evaluate_gnb('diabetes.csv', target_col='Outcome', dataset_name='Diabetes', label_encode=False)

# Breast Cancer (drop id columns)
evaluate_gnb('Breast_Cancer.csv', target_col='diagnosis', dataset_name='Breast Cancer', drop_cols=['id', 'Unnamed: 32'])

# Wine Quality
evaluate_gnb('Wine_Quality.csv', target_col='type', dataset_name='Wine Quality')
