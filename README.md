README
Project Title: MNIST Classification Using Machine Learning
Project Description:
This project demonstrates the classification of handwritten digits using the MNIST dataset. The project applies various machine learning techniques, including Stochastic Gradient Descent (SGD) and k-Nearest Neighbors (k-NN), to achieve high accuracy in digit classification. The notebooks cover essential steps such as data loading, preprocessing, training models, evaluation, and visualization of results.

Files:
MNIST ML.ipynb:

Description: This notebook initiates the MNIST classification project. It starts with loading the MNIST dataset, conducting preliminary exploratory data analysis, and building a Stochastic Gradient Descent (SGD) classifier.
Core Steps:
Data Loading: Loads the MNIST dataset using fetch_openml from sklearn.
Data Preprocessing: The dataset is visualized, and samples are displayed. Data is also split into training and test sets.
Model Building: An SGD classifier is trained for digit classification.
Cross-Validation: The model is evaluated using cross-validation, achieving an accuracy of around 85%.
Scaling: A StandardScaler is applied to improve model accuracy to 90%.
Error Analysis: Confusion matrices are plotted to analyze classification errors visually.
Advanced Techniques: Multi-label classification is introduced using the k-Nearest Neighbors classifier.
Continuation of MNIST.ipynb:

Description: This notebook continues from where the first notebook left off. It introduces multi-output classification and focuses on optimizing the classification pipeline.
Core Steps:
Multi-label Classification: k-Nearest Neighbors is used for multi-label classification, predicting both if the digit is 7 or above and whether it is odd.
Text Classification Preprocessing: Introduces a text preprocessing pipeline using custom transformers to convert text data into a vector format, which is unrelated to the MNIST dataset but highlights a reusable methodology.
Pipeline Implementation: Builds a preprocessing pipeline to convert emails to word counts and further to vectorized form.
Logistic Regression Model: Applies logistic regression to the transformed data and evaluates it using cross-validation.
Evaluation: Outputs precision and recall metrics to assess model performance on the test set.
Dependencies:
Python 3.x
Jupyter Notebook
Libraries:
numpy
matplotlib
sklearn
joblib
scipy
How to Run:
Clone or download the project files.
Install the required dependencies using pip:
Copy code
pip install numpy matplotlib scikit-learn joblib scipy
Open both .ipynb files in Jupyter Notebook or Jupyter Lab.
Run the notebooks sequentially to follow the process of data loading, model training, evaluation, and analysis.
Results:
Accuracy: The SGD classifier achieves an accuracy of ~90% after scaling the features.
Error Analysis: A confusion matrix is plotted to visualize and understand misclassifications.
Multi-label Classification: The k-NN classifier is capable of handling multi-label predictions for complex cases.
Evaluation Metrics: Precision and recall are computed for further insights into model performance.
