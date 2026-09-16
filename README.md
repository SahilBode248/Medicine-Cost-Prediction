# Medicine Cost Predictor

**Author(s):** Sahil Bode  
**Affiliation:** Rashtrasant Tukadoji Maharaj Nagpur University  
**Date:** September 2026  

## Abstract

The **Medicine Cost Predictor** is a machine learning project developed to predict the payable cost of medical treatment using patient, medicine, and treatment-related information. The system considers factors such as age, gender, BMI, chronic condition, medicine name, dosage, quantity, treatment duration, severity, insurance percentage, and discount percentage. A **Random Forest Regression** model is used to predict the treatment cost and achieved an **R² score of approximately 0.93 on held-out test data**. A web-based interface allows users to enter the required treatment details and obtain an estimated cost. The project uses Python for machine learning and HTML, CSS, and JavaScript for the frontend. The trained model is integrated with Python API functions and can be deployed on Vercel. The project demonstrates a practical application of machine learning for estimating medicine and treatment costs from user-provided information.

## Introduction

The cost of medicines and medical treatments can vary depending on factors such as medicine type, dosage, quantity, treatment duration, severity, insurance coverage, and discounts. Estimating the payable cost manually can be difficult. The objective of this project is to develop a machine learning-based **Medicine Cost Predictor** that estimates the payable treatment cost using relevant patient, medicine, and treatment details. The system provides a simple web interface for users to enter information and receive a predicted cost.

## Literature Review

Machine learning techniques are increasingly being applied to healthcare cost prediction and numerical estimation problems. Regression algorithms such as Linear Regression, Decision Trees, Random Forest, and Gradient Boosting can be used to identify relationships between input features and treatment costs. **Random Forest Regression** is effective for handling complex and nonlinear relationships between multiple features. This project applies Random Forest to predict medicine and treatment costs based on patient and treatment information.

## Methodology

The system accepts patient, medicine, and treatment details as input. The input data is preprocessed by handling numerical and categorical features and creating useful features such as **Age_Group, BMI_Category, and Has_Insurance**. A Random Forest Regression model is trained using raw user-input features and valid engineered features while avoiding target-data leakage. The trained model is saved using Joblib. During prediction, the web application sends the entered information to a Python API, which loads the trained model and returns the estimated **Payable_Cost**.

## Implementation

### Programming Languages

- Python
- HTML
- CSS
- JavaScript

### Frameworks/Libraries

- Pandas
- NumPy
- Scikit-learn
- Joblib

### Tools Used

- Google Colab
- Jupyter Notebook
- VS Code
- GitHub
- Vercel

## Results and Discussion

The Random Forest Regression model achieved an **R² score of approximately 0.93 on held-out test data**. The model predicts the payable cost using patient, medicine, dosage, quantity, duration, severity, insurance, and discount information. The web application allows users to enter the required details and receive the predicted medicine/treatment cost through the prediction API.

## Limitation

- Prediction depends on the quality and size of the training dataset.
- The predicted cost may differ from the actual medical bill.
- Uncommon medicines or treatment combinations may produce less accurate predictions.
- Medicine and treatment prices can vary by location, hospital, and time.
- The system is intended for cost prediction and does not provide medical diagnosis or treatment advice.

## Future Scope

- Use larger and more diverse healthcare datasets.
- Add real-time medicine price information.
- Include hospital and region-specific pricing.
- Add prediction ranges or confidence scores.
- Improve the model using advanced machine learning techniques.
- Develop a mobile application.
- Integrate updated medicine and treatment databases.

## Conclusion

The **Medicine Cost Predictor** demonstrates how machine learning can be used to estimate medicine and treatment costs. The Random Forest Regression model uses patient, medicine, and treatment-related information to predict the payable cost. The web-based interface makes the system simple to use, while the Python API provides the prediction functionality. The project demonstrates a practical application of machine learning for healthcare cost estimation.

## References

[1] L. Breiman, "Random Forests," *Machine Learning*, 2001.  
[2] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, 2011.  
[3] Scikit-learn Documentation: https://scikit-learn.org/  
[4] Pandas Documentation: https://pandas.pydata.org/  
[5] Vercel Documentation: https://vercel.com/docs  
[6] GitHub Repository: https://github.com/SahilBode248/Medicine-Cost-Predictor
