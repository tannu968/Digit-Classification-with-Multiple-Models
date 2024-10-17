# Digit-Classification-with-Multiple-Models
This project explores classification techniques on the digits dataset using Logistic Regression, Support Vector Machine (SVM), and Random Forest classifiers. We compare these models' performance using KFold, StratifiedKFold, and cross-validation.

Project Overview
This project aims to:

Train multiple models on the digits dataset.
Use KFold and StratifiedKFold for model evaluation.
Apply cross-validation to measure model performance.

Key Libraries Used
Numpy
Matplotlib
Scikit-learn

Models Used
Logistic Regression
Support Vector Machine (SVC)
Random Forest Classifier

Techniques Employed
KFold Cross-Validation: Splits the dataset into equal parts to train and evaluate the model.
StratifiedKFold: Ensures each fold has the same proportion of classes as the entire dataset.
Cross-Validation: Used to evaluate model performance across different random subsets.


How to Run the Project
git clone https://github.com/your-username/digit-classification.git

pip install -r requirements.txt

python digit_classification.py

Results
The models are compared based on their average accuracy across folds, and the best performing model is identified.

Future Enhancements
Explore more advanced models like Gradient Boosting or XGBoost.
Use hyperparameter tuning for better model performance.

