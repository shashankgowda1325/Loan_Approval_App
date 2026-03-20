import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import pickle

df = pd.read_csv("data.csv")

df = df.drop("Loan_ID", axis=1)

df.fillna(method='ffill', inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

model = GradientBoostingClassifier()
model.fit(X, y)

pickle.dump(model, open("loan_model.pkl", "wb"))

print("Model trained and saved!")
