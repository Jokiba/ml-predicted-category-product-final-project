# Importing the required modules
import pandas as pd
import joblib

# Load the saved model
model = joblib.load(r"c:\Users\10\Desktop\ITAcademy\Introduction to Machine Learning using Python\Taskovi\Taskovi_za_slanje\Jovan_Babic_Task3_PredictedCategoryProduct\ml-predicted-category-product-final-project\model\category_model.pkl")

print("Model loaded successfully!")
print("Type 'close' at any point to stop.\n")

# While loop for user input
while True:
    title = input(" Enter product title: ")
    if title.lower() == "close":
        print("The model is closed.")
        break
    # Important note: Enter the category_title only if it is in the title, otherwise press enter
    category_title = input(" Enter category in title or press enter if no category in title provided: ")
    if category_title.lower() == "close":
        print("The model is closed.")
        break
 
    # Create a DataFrame from input
    user_input = pd.DataFrame([{
        "product_title": title,
        "category_in_title": category_title
    }])
 
    # Predict sentiment
    prediction = model.predict(user_input)[0]
    print(f" Predicted category: {prediction}\n" + "-" * 40)