import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# (1) تحميل وتجهيز البيانات
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
column_names = [
    "Sample_code_number", "Clump_Thickness", "Uniformity_of_Cell_Size",
    "Uniformity_of_Cell_Shape", "Marginal_Adhesion", "Single_Epithelial_Cell_Size",
    "Bare_Nuclei", "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class",
]
df = pd.read_csv(url, names=column_names, na_values="?")
df.dropna(inplace=True)
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'])
df['Class'] = df['Class'].map({2: 0, 4: 1})
X = df.drop(['Sample_code_number', 'Class'], axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (2) بناء وتدريب النموذج باستخدام GridSearchCV و Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(solver='liblinear'))
])
param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# (3) حفظ أفضل نموذج
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'cancer_prediction_model.pkl')
print("تم تدريب النموذج وحفظه بنجاح في ملف 'cancer_prediction_model.pkl'.")