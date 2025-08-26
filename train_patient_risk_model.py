import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib
import onnx
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Simulated hospital dataset
data = pd.DataFrame({
    "age": np.random.randint(18, 90, 1000),
    "blood_pressure": np.random.randint(80, 180, 1000),
    "cholesterol": np.random.randint(150, 300, 1000),
    "hospital_stay_days": np.random.randint(1, 15, 1000),
    "readmitted": np.random.randint(0, 2, 1000)
})

X = data.drop("readmitted", axis=1)
y = data["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML model
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))

# Convert to ONNX for Triton
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type)

with open("models/patient_risk/1/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
