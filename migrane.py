import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


dataset_path = r"C:\Users\mvy48\OneDrive\Desktop\vscodere\migraine.csv"
df = pd.read_csv(dataset_path)


X = df.drop('Type', axis=1)  
y = df['Type']  


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=90)


svm_model = svm.SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)


y_pred = svm_model.predict(X_test)


accuracy_svm = accuracy_score(y_test, y_pred)
precision_svm = precision_score(y_test, y_pred, average='micro')
recall_svm = recall_score(y_test, y_pred, average='micro')
f1_svm = f1_score(y_test, y_pred, average='micro')


print("SVM Model Metrics:")
print(f'Accuracy: {accuracy_svm * 100:.2f}%')
print(f'Precision: {precision_svm:.2f}')
print(f'Recall: {recall_svm:.2f}')
print(f'F1-score: {f1_svm:.2f}')


class_report = classification_report(y_test, y_pred, zero_division=1)
print('Classification Report:')
print(class_report)


class_report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
f1_scores = class_report_df['f1-score'].values
recalls = class_report_df['recall'].values


for i in range(len(label_encoder.classes_)):
    print(f'Class {label_encoder.classes_[i]} - F1-Score: {f1_scores[i]:.2f}, Recall: {recalls[i]:.2f}')


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()





metrics_df = pd.DataFrame({'Precision': precision_svm, 'Recall': recall_svm, 'F1-score': f1_svm}, index=['Overall'])
for i in range(len(label_encoder.classes_)):
    metrics_df = pd.concat([metrics_df,
                            pd.DataFrame({'Precision': precision_score(y_test == i, y_pred == i, average='binary'),
                                          'Recall': recall_score(y_test == i, y_pred == i, average='binary'),
                                          'F1-score': f1_score(y_test == i, y_pred == i, average='binary')},
                                         index=[label_encoder.classes_[i]])])

metrics_df.plot(kind='bar', figsize=(12, 8))
plt.title('Precision, Recall, and F1-score for each class')
plt.xlabel('Class')
plt.ylabel('Score')
plt.show()


user_input = []
for feature in X.columns:
    value = float(input(f"Enter the value for {feature}: "))
    user_input.append(value)


user_input_df = pd.DataFrame([user_input], columns=X.columns)


user_prediction = svm_model.predict(user_input_df)

predicted_class = label_encoder.inverse_transform(user_prediction)[0]

if predicted_class == 1:
    print("Based on the input, the model predicts that you are not likely to have migraine.")
else:
    print("Based on the input, the model predicts that you are likely to have migraine.")


thank_you_message = """
=============================================================
Thank you for using our Migraine Classification Project!
We appreciate your time and hope you find the results valuable.
If you have any questions or feedback, feel free to contact our team.
=============================================================
"""


fig, ax = plt.subplots(figsize=(10, 2))
ax.text(0.5, 0.5, thank_you_message, fontsize=12, va='center', ha='center', color='red')
ax.axis('off')  
plt.show()