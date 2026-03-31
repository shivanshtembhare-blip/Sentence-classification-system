#importing all the modules
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

#reading the data
data = pd.read_csv(r'C:\Users\shivansh tembhare\Desktop\aiml project\synthetic_pop_culture_2000.csv')
X = data['text']
y = data['label']

#splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#converting the text data into numerical format before feeding it to the model. We use CountVectorizer to convert the text into a matrix of token counts.
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

#Training the Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

#Now that the model is trained we can use it to predict the labels for the test data using X_test_vectorized
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

#Caluclating the accuracy and confusion matrix of the model to understand how well it's performing
print(f'Accuracy: {accuracy *100}%')
class_labels = np.unique(y_test)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#Prediction on unseen Data
user_input =input("put your sentence here:")
user_input_vectorized = vectorizer.transform([user_input])
predicted_label = model.predict(user_input_vectorized)
print(f"The input text belongs to the '{predicted_label[0]}' category.")
time.sleep(5)
