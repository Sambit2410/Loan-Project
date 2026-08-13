import pandas as pd
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_val_score,StratifiedGroupKFold 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from dotenv import load_dotenv
import os

load_dotenv()

DATASET_NAME = os.getenv("DATASET_NAME")
TARGET_COL = os.getenv("TARGET_COL")
TEST_SIZE = os.getenv("TEST_SIZE")
RANDOM_STATE = os.getenv("RANDOM_STATE")
MODEL_PATH = os.getenv("MODEL_PATH")


df = pd.read_csv(DATASET_NAME)

X = df.drop(columns=TARGET_COL)

y = df.Loan_Status

y.value_counts(normalize=True)

y.value_counts()

def check_ratio(y):
  d = {
        "Count":y.value_counts(),
        "Percentage":round(y.value_counts(normalize=True),3)
  }
  y_ratio = pd.DataFrame(d)
  return y_ratio

check_ratio(y)

def missing_report(df):
    missing_summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum().sort_values(ascending=False),
            "missing_percent": (df.isna().mean() * 100).sort_values(ascending=False)
        }
    )
    return missing_summary
missing_report(df)

print("Shape Dataset -> ", df.shape)
print("Shape X -> ", X.shape)
print("Shape y -> ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state= RANDOM_STATE,
    stratify=y
)

numerical = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
categorical = ["Loan_ID","Gender","Married","Dependents","Education","Self_Employed","Credit_History","Property_Area"] 

for i in numerical:
    X_train_avg = X_train[i].mean()
    X_train[i] = X_train[i].fillna(X_train_avg)
    X_test[i] = X_test[i].fillna(X_train_avg)

missing_report(X_train)

for i in categorical:
    X_train_mode = X_train[i].mode()[0]
    X_train[i] = X_train[i].fillna(X_train_mode)
    X_test[i] = X_test[i].fillna(X_train_mode)

missing_report(X_train)

train_loan = pd.concat([X_train, y_train], axis=1)
test_loan = pd.concat([X_test, y_test], axis=1)

train_loan.to_csv("train_loan.csv", index=False)
test_loan.to_csv("test_loan.csv", index=False)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

train = pd.read_csv("train_loan.csv")
test = pd.read_csv("test_loan.csv")

X_train = train.drop(columns='Loan_Status') 
y_train = train["Loan_Status"]
X_test = test.drop(columns='Loan_Status') 
y_test = test["Loan_Status"]

print("Shape Dataset -> ", X_train.shape)
print("Shape X -> ", X_test.shape)
print("Shape y -> ", y_train.shape)

ord_bin_cat = ["Education", "Dependents"]
nominal_cat = ["Gender", "Married","Self_Employed","Credit_History","Property_Area"]
to_remove = ["Loan_ID"]

X_train = X_train.drop(columns=to_remove)
X_test = X_test.drop(columns=to_remove)

ord_bin_cat = ["Education", "Dependents"]

label_enc_classes = {}

for i in ord_bin_cat:
    le = LabelEncoder()
    le.fit(X_train[i])
    X_train[i] = le.transform(X_train[i])
    X_test[i] = le.transform(X_test[i])
    
    label_enc_classes[i] = le
    
label_enc_classes

label_enc_classes = {}
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)
label_enc_classes = le

label_enc_classes

y_train = pd.Series(y_train)

nominal_cat = ["Gender", "Married","Self_Employed","Credit_History","Property_Area"]
print("Nominal categorical features are : ", nominal_cat)

# Step 2: Create class instance 
ohe = OneHotEncoder(
    drop='first',
    handle_unknown='ignore',
    sparse_output=False
)

# Step 3 : Fit train data
ohe.fit(X_train[nominal_cat])

# Extracting encoded columns to use as column heading in dataframe
ohe_encoded_cols = ohe.get_feature_names_out()
print(ohe_encoded_cols)

#Step 4: transform train and test data and store the result in variable
train_encoded = ohe.transform(X_train[nominal_cat])
test_encoded = ohe.transform(X_test[nominal_cat])

#Step 5: Convert the step 4 result to dataframe and use columns extracted above
train_df = pd.DataFrame(train_encoded, columns=ohe_encoded_cols, index=X_train.index)
test_df = pd.DataFrame(test_encoded, columns=ohe_encoded_cols, index=X_test.index)
#train_df.head()

#Step 6: Drop nominal_categories created in step 1 from X_train and X_test
X_train = X_train.drop(columns=nominal_cat)
X_test = X_test.drop(columns=nominal_cat)

#Step 7: Concatenate the encoded df and original df - both train and test
X_train = pd.concat([X_train, train_df], axis=1)
X_test = pd.concat([X_test, test_df], axis=1)

X_train.head()

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

check_ratio(y)

smote = SMOTE(random_state=42)
X_train_so, y_train_so = smote.fit_resample(X_train, y_train)

check_ratio(y_train_so)

pipeline = Pipeline([
  ("scaler",StandardScaler()),
  ("model",LogisticRegression())
])
pipeline.fit(X_train,y_train)

train_pred = pipeline.predict(X_train)
test_pred = pipeline.predict(X_test)

prob_train = pipeline.predict_proba(X_train)
prob_test = pipeline.predict_proba(X_test)

skf = StratifiedKFold(
  n_splits=5,
  shuffle=True,
  random_state=42
)

score = cross_val_score(
  pipeline,
  X_train,
  y_train,
  scoring="accuracy",
  cv=skf
)

score.mean()

G = df["Loan_ID"]

sgf = StratifiedGroupKFold(
  n_splits=5,
  shuffle=True,
  random_state=42
)

def print_metrics(title, y_true, y_pred):
    acc = accuracy_score(y_true,y_pred )   # (test_labels, prediction_labels)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(title)
    print(f"  Accuracy: {acc:.2%}")
    print(f"  Precision: {prec:.2%}")
    print(f"  Recall: {rec:.2%}")
    print(f"  F1-score: {f1:.2%}")

print_metrics("TRAIN METRICS", y_train, train_pred)
print_metrics("TEST METRICS", y_test, test_pred)

cm = confusion_matrix(y_train, train_pred)

print("Confusion Matrix")
print(cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"],fmt=".0f")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix (Train Data)")
plt.show()

cm = confusion_matrix(y_test, test_pred)

print("Confusion Matrix")
print(cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix (Test Data)")
plt.show()

print(df.head())

dump(pipeline, MODEL_PATH)