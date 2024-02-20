# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
# Replace 'dataset.csv' with your actual dataset file path
data = pd.read_csv('dataset.csv')

# Perform any necessary data preprocessing
# For example, handle missing values, encode categorical variables, etc.

# Split data into features (X) and labels (y)
X = data.drop('lung_disease_label', axis=1)  # Assuming 'lung_disease_label' is the name of the target variable
y = data['lung_disease_label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model (Random Forest Classifier in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# You can also print more detailed evaluation metrics
print(classification_report(y_test, predictions))

# Now, you can use this trained model to make predictions on new data
# For example:
# new_data = ... # Load or input new data
# new_predictions = model.predict(new_data)
