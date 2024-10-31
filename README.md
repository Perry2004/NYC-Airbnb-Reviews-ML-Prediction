# NYC Airbnb Reviews ML Prediction

This repository presents a machine learning project that predicts the number of reviews per month for Airbnb listings in New York City. The project is inspired by coursework from UBC CPSC 330.

## Project Overview

The goal of this project is to analyze the NYC Airbnb dataset and develop a predictive model for the `reviews_per_month` variable, providing insights for hosts and potential guests regarding listing popularity.

## Dataset

The analysis utilizes an Airbnb dataset featuring various attributes. The dataset includes 48,895 instances and 16 features. 
The target variable for prediction is `reviews_per_month`.

The dataset is available on Kaggle: [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data). 

## Data Processing
- Imputation: instances with missing values in the target column are removed, numerical features are imputed with the mean, and categorical features are imputed with "missing".
- Standardization: numerical features are standardized using the `StandardScaler`.
- Categorical Encoding: categorical features are encoded using the `OneHotEncoder`.
- Text encoding: text features are encoded using the `CountVectorizer`.
- Discretization: the location features are discretized into bins using the `KBinsDiscretizer`.

## Model Development

The project includes the implementation of various machine learning models to predict `reviews_per_month`.   
Included models are:
- Linear Regression (`Ridge` for regularization)
- Decision Tree Regressor
- Support Vector Regressor with RBF kernel
- XGBoost Regressor ensemble model

## Model Evaluation
- Surprisingly, the untuned decision tree regressor performed very well on both the validation and test sets.
- The decision tree model is highly overfitting as the training error is extremely close to zero. Thus, there's a possibility that the model can be improved by reducing the complexity of the model. This can be done by tuning the hyperparameters more comprehensively.
- The performance of the SVM and XGBoost models is similar with some random variation. But the SVM model is much slower than the XGBoost model and is more time-consuming for comprehensive hyperparameter tuning.  
- The linear model performs poorly, likely due to the non-linear relationship between the features and the target variable.

## Running the Project

To run the project locally:
1. Clone the repository.
2. Download the dataset from the Kaggle link above and place it in the `./data` directory.
3. Install the required packages using `pip install -r requirements.txt`.
4. Run the Jupyter notebook `NYC_Airbnb_Reviews_ML_Prediction.ipynb`.

## Possible Improvements
- More advanced feature engineering, such as feature interactions and polynomial features.
- Hyperparameter tuning for a larger range of hyperparameters and more iterations.
- Better encoding for text features. 

## License & Contact
This project is distributed under the MIT License. See LICENSE for more information.   
For inquiries or feedback, please email [perryzhu2004@outlook.com](mailto:).   
Contributions are welcome! 

This project is inspired by coursework from UBC CPSC 330 Applied Machine Learning. More details can be found in the [course repository](https://github.com/UBC-CS/cpsc330-2024W1?tab=readme-ov-file)
