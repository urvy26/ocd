from pyexpat import model

from django.shortcuts import render
import pandas as pd
import numpy as np
from django.shortcuts import render
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import  VotingClassifier
from scipy.sparse import issparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    # showing nan

    data = pd.read_csv("E:\ocd numeric 2.csv")
    X = data.drop(columns=['Medications'], axis=1)
    y = data['Medications']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()
    gb_classifier = GradientBoostingClassifier()

    # Create a voting classifier with soft voting
    voting_classifier = VotingClassifier(estimators=[
        ('rf', rf_classifier),
        ('gb', gb_classifier)

    ], voting='soft')
    # Train the voting classifier
    voting_classifier.fit(X_train, y_train)

    val1 = float(request.GET.get('n1', 0))
    val2 = float(request.GET.get('n2', 0))
    val3 = float(request.GET.get('n3', 0))
    val4 = float(request.GET.get('n4', 0))
    val5 = float(request.GET.get('n5', 0))
    val6 = float(request.GET.get('n6', 0))
    val7 = float(request.GET.get('n7', 0))


    pred = voting_classifier.predict([[val1, val2, val3, val4, val5, val6, val7]])
    result1 = ""
    if pred == [1]:
        result1= "positive"
    else:
        result1 = "Negative"


    return render(request, "predict.html",{"result2":result1})



    # data = pd.read_csv("E:\ocdsim.csv")
    # X = data.drop(columns=['Medications'], axis=1)
    # y = data['Medications']
    #
    # # Handle missing values
    # X.fillna(method='ffill', inplace=True)
    # y.fillna(method='ffill', inplace=True)
    #
    # # Check unique classes in the target variable
    # unique_classes = y.unique()
    #
    # if len(unique_classes) < 2:
    #     # Apply SMOTE to handle class imbalance
    #     smote = SMOTE(random_state=42)
    #     X_resampled, y_resampled = smote.fit_resample(X, y)
    #     X = X_resampled
    #     y = y_resampled
    #
    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # categorical_features = ['Gender', 'Ethnicity', 'Marital Status', 'Education Level', 'Previous Diagnoses',
    #                         'Family History of OCD', 'Obsession Type', 'Compulsion Type',
    #                         'Depression Diagnosis', 'Anxiety Diagnosis']
    #
    # # Create a column transformer for one-hot encoding
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('cat', OneHotEncoder(), categorical_features)
    #     ], remainder='passthrough')  # Remainder to handle non-categorical features
    #
    # # Create a pipeline with Naive Bayes Classifier
    # nb_model = Pipeline([
    #     ('preprocessor', preprocessor),
    #     ('classifier', MultinomialNB())
    # ])
    #
    # # Train the Naive Bayes model
    # nb_model.fit(X_train, y_train)
    #
    # # Debugging request parameters
    # val1 = request.GET.get('n1', '')  # Assuming n1 is a string
    # val2 = request.GET.get('n2', '')  # Assuming n2 is a string
    # val3 = request.GET.get('n3', '')  # Assuming n3 is a string
    # val4 = request.GET.get('n4', '')  # Assuming n4 is a string
    # val5 = request.GET.get('n5', '')  # Assuming n5 is a string
    # val6 = request.GET.get('n6', '')  # Assuming n6 is a string
    # val7 = request.GET.get('n7', '')  # Assuming n7 is a string
    # val8 = request.GET.get('n8', '')  # Assuming n8 is a string
    # val9 = request.GET.get('n9', '')  # Assuming n9 is a string
    # val10 = request.GET.get('n10', '')  # Assuming n10 is a string
    #
    # # Create a DataFrame with input values
    # input_data = pd.DataFrame({
    #     'Gender': [val1],
    #     'Ethnicity': [val2],
    #     'Marital Status': [val3],
    #     'Education Level': [val4],
    #     'Previous Diagnoses': [val5],
    #     'Family History of OCD': [val6],
    #     'Obsession Type': [val7],
    #     'Compulsion Type': [val8],
    #     'Depression Diagnosis': [val9],
    #     'Anxiety Diagnosis': [val10]
    # })
    #
    # # Preprocess input data
    # input_data = preprocessor.transform(input_data)
    #
    # # Make prediction
    # pred = nb_model.predict(input_data)
    #
    # # Convert prediction to result
    # result = "Positive" if 'Yes' in pred else "Negative"
    #
    # return render(request, "predict.html", {"result2": result})



    # showing male
    # data = pd.read_csv("E:\ocdsim.csv")
    #
    # # Assuming 'Medications' is the target variable
    # y = data['Medications']
    # X = data.drop(columns=['Medications'], axis=1)
    #
    # # Handle missing values
    # X.fillna(method='ffill', inplace=True)
    #
    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # # Assuming these are the categorical features
    # categorical_features = ['Gender', 'Ethnicity', 'Marital Status', 'Education Level', 'Previous Diagnoses',
    #                         'Family History of OCD', 'Obsession Type', 'Compulsion Type',
    #                         'Depression Diagnosis', 'Anxiety Diagnosis']
    #
    # # Create a column transformer for one-hot encoding
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('imputer', SimpleImputer(strategy='most_frequent'), categorical_features),
    #         ('cat', OneHotEncoder(drop='first'), categorical_features)
    #     ])
    #
    # # Create a pipeline with Multinomial Naive Bayes Classifier
    # nb_model = Pipeline([
    #     ('preprocessor', preprocessor),
    #     ('classifier', MultinomialNB())
    # ])
    #
    # # Train the Multinomial Naive Bayes model
    # nb_model.fit(X_train, y_train)
    #
    # # Assuming these are the request parameters
    # val1 = request.GET.get('n1', '')  # Assuming n1 is a string
    # val2 = request.GET.get('n2', '')  # Assuming n2 is a string
    # val3 = request.GET.get('n3', '')  # Assuming n3 is a string
    # val4 = request.GET.get('n4', '')  # Assuming n4 is a string
    # val5 = request.GET.get('n5', '')  # Assuming n5 is a string
    # val6 = request.GET.get('n6', '')  # Assuming n6 is a string
    # val7 = request.GET.get('n7', '')  # Assuming n7 is a string
    # val8 = request.GET.get('n8', '')  # Assuming n8 is a string
    # val9 = request.GET.get('n9', '')  # Assuming n9 is a string
    # val10 = request.GET.get('n10', '')  # Assuming n10 is a string
    #
    # # Create a DataFrame with input values
    # input_data = pd.DataFrame({
    #     'Gender': [val1],
    #     'Ethnicity': [val2],
    #     'Marital Status': [val3],
    #     'Education Level': [val4],
    #     'Previous Diagnoses': [val5],
    #     'Family History of OCD': [val6],
    #     'Obsession Type': [val7],
    #     'Compulsion Type': [val8],
    #     'Depression Diagnosis': [val9],
    #     'Anxiety Diagnosis': [val10]
    # })
    #
    # # Handle missing values
    # input_data.fillna(method='ffill', inplace=True)
    #
    # # Preprocess input data
    # input_data = preprocessor.transform(input_data)
    #
    # # Make prediction
    # pred = nb_model.predict(input_data)
    #
    # # Convert prediction to result
    # result = "Positive" if 'Yes' in pred else "Negative"
    #
    # return render(request, "predict.html", {"result2": result})

    # data = pd.read_csv("E:\ocdsim.csv")
    #
    # # Handle missing values
    # data.fillna(method='ffill', inplace=True)
    #
    # # Split the data into features (X) and target (y)
    # X = data.drop(columns=['Medications'], axis=1)
    # y = data['Medications']
    #
    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # categorical_features = ['Gender', 'Ethnicity', 'Marital Status', 'Education Level', 'Previous Diagnoses',
    #                         'Family History of OCD', 'Obsession Type', 'Compulsion Type',
    #                         'Depression Diagnosis', 'Anxiety Diagnosis']
    #
    # # Create a column transformer for one-hot encoding
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('cat', OneHotEncoder(drop='first'), categorical_features)
    #     ])
    #
    # # Create a pipeline with Naive Bayes Classifier
    # nb_model = Pipeline([
    #     ('preprocessor', preprocessor),
    #     ('classifier', GaussianNB())
    # ])
    #
    # # Train the Naive Bayes model
    # nb_model.fit(X_train, y_train)
    #
    # # Get input data from request parameters
    # val1 = request.GET.get('n1', '')  # Assuming n1 is a string
    # val2 = request.GET.get('n2', '')  # Assuming n2 is a string
    # val3 = request.GET.get('n3', '')  # Assuming n3 is a string
    # val4 = request.GET.get('n4', '')  # Assuming n4 is a string
    # val5 = request.GET.get('n5', '')  # Assuming n5 is a string
    # val6 = request.GET.get('n6', '')  # Assuming n6 is a string
    # val7 = request.GET.get('n7', '')  # Assuming n7 is a string
    # val8 = request.GET.get('n8', '')  # Assuming n8 is a string
    # val9 = request.GET.get('n9', '')  # Assuming n9 is a string
    # val10 = request.GET.get('n10', '')  # Assuming n10 is a string
    #
    # # Create a DataFrame with input values
    # input_data = pd.DataFrame({
    #     'Gender': [val1],
    #     'Ethnicity': [val2],
    #     'Marital Status': [val3],
    #     'Education Level': [val4],
    #     'Previous Diagnoses': [val5],
    #     'Family History of OCD': [val6],
    #     'Obsession Type': [val7],
    #     'Compulsion Type': [val8],
    #     'Depression Diagnosis': [val9],
    #     'Anxiety Diagnosis': [val10]
    # })
    #
    # # Preprocess input data
    # input_data = preprocessor.transform(input_data)
    #
    # if issparse(input_data):
    #     input_data = input_data.toarray()
    #
    # try:
    #     # Make predictions on input data
    #     pred = nb_model.predict(input_data)
    #
    #     # Convert prediction to result
    #     result = "Positive" if 'Yes' in pred else "Negative"
    #
    # except ValueError as e:
    #     # Handle the case where the input data cannot be converted to float
    #     result = "Error: Input data could not be processed"
    #
    # return render(request, "predict.html", {"result2": result})