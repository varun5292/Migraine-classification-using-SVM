This project is a Migraine Classification system utilizing machine learning techniques, primarily Support Vector Machine (SVM). Here's a breakdown of each step:

Importing Libraries: Import necessary libraries like Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn modules.

Loading Dataset: Read the dataset from the provided CSV file path using Pandas.

Data Preprocessing:

Separate features (X) and target variable (y).
Encode categorical labels into numerical format using LabelEncoder.
Train-Test Split: Split the dataset into training and testing sets using train_test_split() function.

Building SVM Model:

Initialize an SVM model with a linear kernel and enable probability estimates.
Train the SVM model using the training data.
Model Evaluation:

Predict the target variable for the test set using the trained model.
Calculate various evaluation metrics such as accuracy, precision, recall, and F1-score using scikit-learn metrics functions.
Print the classification report providing detailed metrics for each class.
Confusion Matrix Visualization: Visualize the confusion matrix using Seaborn's heatmap.

Metrics Visualization:

Plot precision, recall, and F1-score for each class.
Calculate overall metrics and plot them.
User Input and Prediction:

Prompt the user to input values for each feature.
Create a DataFrame with the user input and predict the class using the trained model.
Inverse transform the predicted class to get the meaningful label (migraine or not migraine).
Thank You Message: Display a thank you message using Matplotlib with a customized message.
The project involves data visualization using Matplotlib and Seaborn libraries for enhanced understanding of model results and user interaction.
