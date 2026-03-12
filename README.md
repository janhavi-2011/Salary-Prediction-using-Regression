# Salary Prediction Using Regression

## Project Title
Salary Prediction Using Regression

## Author
**Name:** Janhavi Maurya  
**Batch:** 8

---

## Project Description
Salary prediction is an important task for organizations to determine fair employee compensation. This project uses **Machine Learning Regression techniques** to predict employee salaries based on several factors such as age, gender, education level, job title, and years of experience.

The project includes **data preprocessing, model training, model evaluation, visualization, and an interactive Streamlit dashboard** that allows users to predict salaries based on input features.

---

## Objectives
- To build a machine learning model that predicts employee salary.
- To analyze the impact of features like age, gender, education, job title, and experience on salary.
- To compare different regression models.
- To develop an interactive dashboard for salary prediction.

---

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit

---

## Dataset Description
The dataset contains employee information used for salary prediction.

### Features in the Dataset
- Age
- Gender
- Education Level
- Job Title
- Years of Experience

### Target Variable
- Salary

---

## Methodology

### 1. Data Preprocessing
- Removed missing values using `dropna()`
- Separated features (X) and target variable (Salary)
- Converted categorical variables using **One-Hot Encoding**
- Split dataset into **training and testing data**

### 2. Machine Learning Models Used
- Linear Regression
- Random Forest Regressor

### 3. Model Training
The dataset was divided into:
- **80% Training Data**
- **20% Testing Data**

Models were trained using the **Scikit-learn library**.

### 4. Model Evaluation
Model performance was evaluated using:
- **R² Score**
- **Mean Squared Error (MSE)**

---

## Data Visualization
The following visualizations were created to understand data patterns:

- Salary vs Years of Experience
- Actual Salary vs Predicted Salary

These graphs help analyze relationships between features and salary.

---

## Streamlit Dashboard
An interactive **Streamlit dashboard** was created with the following features:

- Sidebar navigation
- Dataset preview
- Interactive salary visualization
- Model performance comparison
- Salary prediction interface

Users can enter their details such as age, education, job title, gender, and experience to get a predicted salary.

---

## Project Workflow
1. Data Collection
2. Data Preprocessing
3. Feature Encoding
4. Model Training
5. Model Evaluation
6. Data Visualization
7. Streamlit Dashboard Development

---

## Results
The machine learning models were able to predict employee salaries based on the input features.  
Both **Linear Regression** and **Random Forest Regressor** produced meaningful predictions, and the results were visualized through graphs and the Streamlit dashboard.

---

## Future Improvements
- Use larger real-world datasets
- Add more machine learning models
- Improve model accuracy using hyperparameter tuning
- Deploy the project online

---

## Conclusion
This project demonstrates how machine learning can be used to predict employee salaries using regression techniques. It also shows how interactive dashboards can make data analysis and prediction easier for users.


