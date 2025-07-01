import streamlit as st
from PIL import Image
import pandas as pd
import base64


def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def set_title_color(color):
    st.markdown(
        f"""
        <style>
        h1 {{
            color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True)


def colored_text(text, color):
    st.markdown(f"<span style='color: {color};'>{text}</span>",
                unsafe_allow_html=True)


def welcome():
    set_background('Data_mining_pr_logo_rouge.png')
    set_title_color("white")
    st.title("Data Loading Support")
    authors = "By:\n Alan Nonso(171417)\n Antoine()\n Irenée()\n Sabarie()"
    colored_text(authors, "white")
    welcome_text = "Welcome to Data_File Traitor (DaFiT) !"
    line1 = "We are delighted to have you here.\n "
    line2 = "This application is designed to help you loading and analyzing data in a very deep eye."
    line3 = "Fell free to explore our content in the 'Navigation section'..."
    colored_text(welcome_text, 'white')
    colored_text(line1, 'white')
    colored_text(line2, 'white')
    colored_text(line3, 'white')


def data_obs_intro():
    set_background('Data_mining_pr_logo.png')
    set_title_color("white")
    st.title('Data observation farm')
    st.write("""Welcome to the Data Observation section! 
                    Here, you can explore various aspects of the dataset through interactive visualizations.

                    How it works:

                    1.Select a Category: Use the dropdown menu to choose a specific category (column) from the dataset,
                    2.See the Histogram: Once you select a category, a histogram will be generated and displayed. 
                    This histogram provides a visual representation of the distribution of values within the selected 
                    category.
                    3.Analyze the Data: The histogram helps you understand the frequency and distribution of data points
                    in the chosen category, making it easier to identify patterns, trends, and outliers 

                    Now let's delimit the data you want to observe...:""")


def formatting_data_intro():
    set_background('Data_mining_pr_logo_bleu.png')
    set_title_color("white")
    st.title('Data formatting farm')
    st.write("""Welcome to the formatting farm, this is where you recook the data's format""")


def uploading():
    data = None
    try:
        file = st.file_uploader("Upload your file here.", type=["csv", "xlsx"])
        # Si un fichier a été téléchargé

        if file is not None:
            # Détermine le type de fichier et l'ouvrir en consequence

            if file.type == "text/csv":
                data = pd.read_csv(file)
                st.success("""FIle successfully uploaded !
                              The uploaded file is a CSV_type""")

            elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                data = pd.read_excel(file)
                st.success("""FIle successfully uploaded !
                              The uploaded file is an Excel_type""")
            return data
        else:
            st.write("Waiting for an uploaded file")

    except Exception as e:
        st.write(f"""Sorry!
                     Something went rong...
                     Error:{e}""")


def ml_regressions_intro():
    set_background('Data_mining_pr_logo_indigo.png')
    set_title_color("white")
    st.title("Machine Learning/ Linear Models")
    with st.expander("Section presentation"):
        st.write(""" 
                 📊 Machine Learning Section: Automated Prediction with Regression
                    Discover how to model complex relationships in your data and make reliable predictions!
                    
                    🔹 Key Features
                    Two Powerful Algorithms:
                    
                 >Linear Regression: Predicts numerical values (e.g., prices, temperatures).
                    
                 >Logistic Regression: Classifies binary outcomes (e.g., Yes/No, Fraud/Legitimate).
                    
                 Smart Data Preparation:
                    
                    Automatic or manual encoding of categorical variables.
                    
                    Numerical data normalization (Standardization or MinMax Scaling).
                    
                 Rigorous Model Evaluation:
                    
                 >For Linear Regression:
                    
                    📉 R² (Coefficient of Determination): Measures prediction accuracy.
                    
                    📏 MSE (Mean Squared Error): Quantifies average prediction error.
                    
                    📌 P-values: Identifies statistically significant variables.
                    
                 >For Logistic Regression:
                    
                    🎯 Confusion Matrix: Visualizes true/false positives/negatives.
                    
                    📊 ROC Curve & AUC: Evaluates classification performance.
                    
                    🔍 Precision, Recall, F1-score: Detailed prediction analysis.
                    
                 Test Your Own Data:
                    
                    Input custom values to generate instant predictions.
                    
                 🚀 How to Use This Section?
                    >Select Your Variables:
                    
                        Choose predictors (input features) and the target (variable to predict).
                    
                    >Prepare Your Data:
                    
                        The tool guides you through encoding and normalization if needed.
                    
                    >Train the Model:
                    
                        The model trains and displays real-time performance metrics.
                    
                    >Interpret Results:
                    
                        Clear explanations help you understand each evaluation metric.
                    
                    💡 Pro Tip: Use the expandable sections (📂) to explore technical details and interpretation guides!
                    
                 📌 Why These Models?
                    >Simplicity & Efficiency: Ideal for beginners and experts alike.
                    
                    >Transparency: Every step is explained (coefficients, errors, etc.).
                    
                    >Flexibility: Works with numerical or categorical data.
                    
                 Ready to predict the future? Select your data and click "Kick-Start"! """)
