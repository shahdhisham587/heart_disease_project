# Heart Disease Detection System:

# This project aims to develop a Heart Disease Detection System using both a rule-based expert system (Experta) and a machine learning model (Decision Tree Classifier in Scikit-Learn). The system analyzes patient health indicators to predict heart disease risk. The project includes data preprocessing, visualization, model training, evaluation, and a Streamlit UI for user interaction.


# The system consists of:
	# Data Preprocessing: Cleaning, encoding, normalizing, and selecting features.
	#	Data Visualization: Exploring relationships using statistical summaries and plots.
	#	Rule-Based Expert System (Experta): A knowledge-based inference system for risk assessment.
	#	Decision Tree Classifier (Scikit-Learn): A machine learning model trained to predict heart disease.
	#	Performance Comparison: Evaluating accuracy and explainability of both approaches.
	#	Streamlit UI: Interactive web interface for easy user input and result visualization.

# Table of Contents:
	# Features:
	#	Technologies Used
	#	Folder Structure
	#	Installation
	#	Usage
	#	Project Workflow
	#	Results and Evaluation
	#	Contributors

# Technologies Used:
	# Python (Data Processing & Model Training)
	#	Pandas, NumPy (Data Handling)
	#	Seaborn, Matplotlib (Visualization)
	#	Experta (Rule-Based Expert System)
	#	Scikit-Learn (Decision Tree Model)
 

# Heart_Disease_Detection/
│── data/ 
│   ├── raw_data.csv            # Original dataset
│   ├── cleaned_data.csv        # Processed dataset
│
│── notebooks/                  # Jupyter Notebooks for analysis
│   ├── data_analysis.ipynb     # Exploratory data analysis
│   ├── model_training.ipynb    # Model training and evaluation
│
│── rule_based_system/          # Rule-based system using Experta
│   ├── rules.py                # Defined rules for heart disease prediction
│   ├── expert_system.py        # Rule engine implementation
│
│── ml_model/                   # Decision Tree model implementation
│   ├── train_model.py          # Training script
│   ├── predict.py              # Model inference script
│
│── utils/                      # Helper functions
│   ├── data_processing.py      # Data cleaning and feature processing
│
│── reports/                    # Model comparison reports
│   ├── accuracy_comparison.md  # Evaluation results
│
│── ui/                         # Streamlit UI for interaction
│   ├── app.py                  # User-friendly interface
│
│── README.md                   # Project documentation
│── requirements.txt            # Dependencies list


# Data Preprocessing:
#Steps Involved:
	# Load Dataset: Read heart disease dataset using pandas.
	#	Handle Missing Values: Fill missing values with mean/median or remove rows.
	#	Normalize Data: Use MinMaxScaler for scaling numeric features.
	#	Encode Categorical Features: Convert categorical variables using One-Hot Encoding.
	# Feature Selection: Identify important features through correlation analysis.
	#	Save Processed Data: Store cleaned dataset as cleaned_data.csv.


# Data Visualization:
#Key Insights are derived using Seaborn & Matplotlib:
	#	Statistical Summary: View distributions of key features.
	#	Correlation Heatmap: Understand feature relationships.
	#	Histograms & Boxplots: Identify outliers and trends.
	#	Feature Importance Plot: Rank influential features for prediction.


# Rule-Based Expert System (Experta):
 A knowledge-based system applies predefined rules to assess heart disease risk.


# Machine Learning Model (Decision Tree Classifier):
 A Decision Tree Classifier is trained to predict heart disease risk.


# Evaluating Performance
The model is evaluated using:
	#       Accuracy
	#	Precision
	#	Recall
	#	F1-score


# Performance Comparison:
The rule-based system and Decision Tree model are evaluated using:
	#	Validation Set Testing: Assess performance on unseen data.
	#	Accuracy Comparison: Analyze key metrics.
	#	Explainability: Understand decision rules vs. machine learning logic.

 
