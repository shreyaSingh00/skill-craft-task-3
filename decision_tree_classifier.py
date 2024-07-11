# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset from local file
file_path = r'C:\Users\singh\OneDrive\Desktop\skill craft task\bank-additional-full.csv'  # Update this with your local file path
data = pd.read_csv(file_path, sep=';')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Preprocess the data
# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, drop_first=True)

# Split the dataset into features and target variable
X = data.drop('y_yes', axis=1)  # Assuming 'y' is the target column and we want to predict 'yes'
y = data['y_yes']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()
