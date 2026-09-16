Medicine Treatment Cost Estimator

Author(s): Sahil Bode
Affiliation: Rashtrasant Tukadoji Maharaj Nagpur University
Date: September 2026

Abstract

The Medicine Treatment Cost Estimator is a machine learning project designed to predict the payable cost of medical treatment based on patient, medicine, and treatment-related details. The system uses a Random Forest Regression model trained on features such as age, BMI, medicine name, dosage, quantity, treatment duration, severity, insurance percentage, and discount percentage. The model achieved an R² score of approximately 0.93 on held-out test data, indicating strong predictive performance. A web-based interface allows users to enter treatment information and receive an estimated payable cost. The application uses a static frontend with Python-based serverless API functions and can be deployed on Vercel.

Introduction

Medical treatment costs can vary depending on medicines, dosage, treatment duration, patient characteristics, insurance coverage, and discounts. Estimating the payable amount manually can be difficult. This project aims to develop a machine learning-based system that estimates treatment cost from user-provided information. It provides a simple web interface where users can enter patient and treatment details and receive an estimated payable cost.

Literature Review

Machine learning regression techniques are widely used for predicting numerical values such as healthcare expenses. Algorithms including Linear Regression, Decision Trees, Random Forest, and Gradient Boosting can be applied to healthcare cost prediction. Random Forest is useful because it can capture nonlinear relationships between multiple patient and treatment features. Feature engineering can further improve prediction performance by creating meaningful features from available input data.

Methodology

The system accepts patient, medicine, and treatment details as input. The data is preprocessed using numerical and categorical feature handling, along with engineered features such as Age_Group, BMI_Category, and Has_Insurance. A Random Forest Regression model is trained using only raw user-input features and valid engineered features to avoid data leakage. The trained model is stored using Joblib and loaded by the Python API during prediction. The web frontend sends user inputs to the API, which returns the estimated Payable_Cost.

Implementation
Programming Languages
Python
HTML
CSS
JavaScript
Frameworks/Libraries
Pandas
NumPy
Scikit-learn
Joblib
Tools Used
Google Colab / Jupyter Notebook
VS Code
GitHub
Vercel
Results and Discussion

The Random Forest model achieved an R² score of approximately 0.93 on the held-out test data. The system predicts the payable treatment cost using patient, medicine, dosage, quantity, duration, severity, insurance, and discount information. The web application provides predictions through a simple user interface and Python serverless API.

Limitation
Prediction accuracy depends on the quality and representativeness of the training dataset.
The model provides an estimated cost and not an actual medical bill.
Unusual medicines or treatment combinations may produce less reliable predictions.
Healthcare prices can vary between hospitals, regions, and time periods.
The system does not provide medical diagnosis or treatment recommendations.
Future Scope
Add larger and more diverse healthcare datasets.
Include hospital and location-specific pricing.
Improve the model using advanced ensemble and deep learning techniques.
Add prediction confidence or estimated cost ranges.
Develop a mobile application.
Integrate regularly updated medicine and treatment prices.
Conclusion

The Medicine Treatment Cost Estimator demonstrates how machine learning can be applied to estimate medical treatment expenses. By using patient, medicine, and treatment-related information with a Random Forest regression model, the system provides an automated estimate of payable cost. The web-based implementation makes the prediction system accessible through a simple user interface and demonstrates a practical application of machine learning in healthcare cost estimation.

References

[1] L. Breiman, "Random Forests," Machine Learning, 2001.
[2] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, 2011.
[3] Scikit-learn Documentation.
[4] Pandas Documentation.
[5] Vercel Documentation.
