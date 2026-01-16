# Importing the required modules
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib

# Loading a clean csv file from the data folder
df_clean = pd.read_csv(r"c:\Users\10\Desktop\ITAcademy\Introduction to Machine Learning using Python\Taskovi\Taskovi_za_slanje\Jovan_Babic_Task3_PredictedCategoryProduct\ml-predicted-category-product-final-project\data\products_clean.csv")

# Converting columns to the appropriate type
df_clean[['category_label', 'category_in_title']] = df_clean[['category_label', 'category_in_title']].astype('category')

# Division of columns into input and output features
X = df_clean[["product_title", "category_in_title"]]
y = df_clean["category_label"]

# Defining columns for ColumnTransformer
text_feature = "product_title"
categorical_feature = ["category_in_title"]

# ColumnTransformer (TF-IDF + OneHot)
preprocessor = ColumnTransformer(
    transformers=[
        # TF-IDF for product name
        ("title_tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=5,
            stop_words="english"
        ), text_feature),

        # One-Hot for category_in_title
        ("category_ohe", OneHotEncoder(
            handle_unknown="ignore"
        ), categorical_feature)
    ]
)

pipeline = Pipeline(
    steps=[
           ("preprocessing", preprocessor),
           ("classifier", LinearSVC())
        ]
)

# Train the model on the entire dataset
pipeline.fit(X, y)

# Save the model to a file
joblib.dump(pipeline, r"c:\Users\10\Desktop\ITAcademy\Introduction to Machine Learning using Python\Taskovi\Taskovi_za_slanje\Jovan_Babic_Task3_PredictedCategoryProduct\ml-predicted-category-product-final-project\model\category_model.pkl")

print("Model trained and saved as 'category_model.pkl'")