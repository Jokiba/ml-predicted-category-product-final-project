# 🧠 Predicted Category Product ML Final Project (Complete Pipeline)

This repository contains a complete machine learning pipeline for **predicted category** of product titles using Python and scikit-learn in Google Colab and Visual Studio Code.

In this project we developed and demonstrated all phases of a machine learning workflow — from raw data to a ready-to-use trained model.

---

## 📦 Project Structure

├── data/

│ └── products.csv # dataset

│ └── products_clean.csv # cleaned and prepared dataset for model

├── notebooks/

│ └── product_category_analysis.ipynb # EDA (Exploratory Data Analysis)

│ └── product_category_preprocessing.ipynb # Preprocessing and feature engineering

│ └── product_category_model.ipynb # Process creating models


├── src/

│ └── train_model.py # Script for training and saving the model

│ └── test_model.py # Script for testing saved model

├── .gitignore  # Files and folders to ignore

├── README.md  # Brief content and the most important information about the project

---

## ✅ What We Did in This Module

Throughout this module, we covered all major steps of a real-world ML project:

### 1. Project Setup
- Created a new GitHub repository
- Defined project folder structure
- Uploaded raw dataset

### 2. Data Exploration
- Loaded and analyzed a large dataset with a various product information
- Used `matplotlib` and `seaborn` for visualizations
- Investigated distribution of category and text characteristics

### 3. Data Cleaning & Preprocessing
- Removed missing values
- Standardized product category labels (fridge freezers / phone	/ washing machines /etc.)
- Creating a new column - category in title for better model predictions
- Explored correlation between various columns and category_label (target variable)
- Deleting unnecessary columns and converting columns to the appropriate type

### 4. Model Training & Evaluation
- Compared multiple ML models (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM)
- Used `ColumnTransformer` and `Pipeline` for unified preprocessing
- Evaluated using precision, recall, F1-score, accuracy, and confusion matrix

### 5. Final Model Training
- Trained final model on full dataset
- Saved the pipeline using `joblib` to `category_model.pkl`

### 6. Inference & Usage
- Loaded saved model
- Built an interactive interface for predicting category of new products title
- Enabled real-time testing via console input

---

## 🚀 How to Use

### 🔧 Train the Model
```cmd
cd src  # put whole destination path when using the training model file
Create folder named 'model' for category_model.pkl in the root directory.
python train_model.py
```

This will create a file called category_model.pkl in the root directory.

### 🔍 Run Inference
Use the interactive script (model_test.py) to classify new products title using the trained model.
