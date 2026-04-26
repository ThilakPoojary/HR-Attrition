# ================================
# PLAN:
# 1. Load dataset
# 2. Clean data
# 3. Encode categorical variables
# 4. Split data
# 5. Train RandomForest
# 6. Save model + columns
# ================================

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv("HR-Employee-Attrition.csv")

# 2. Drop useless columns
df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis=1, inplace=True)

# 3. Encode target
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

# 4. One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# 5. Split data
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Save model & columns
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

print("Model trained & saved successfully!")