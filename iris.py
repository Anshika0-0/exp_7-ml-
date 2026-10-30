# backend
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle #library to save the model(pickeling)

df = pd.read_csv('Iris (1).csv')



df = df.drop("Id", axis=1)
x = df.drop("Species", axis=1)
y = df["Species"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
iris_model = LogisticRegression()
iris_model.fit(x_train, y_train)
y_pred = iris_model.predict(x_test)
print("predicted values for first sample:", y_pred[0])
accuracy = accuracy_score(y_pred, y_test)
print("accuracy of the model is:", accuracy)

# Save the trained model
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(iris_model, file)

print("Model saved as iris_model.pkl")