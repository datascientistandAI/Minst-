
# MNIST Classification Using Machine Learning

## Project Overview
This project demonstrates the classification of handwritten digits from the MNIST dataset using machine learning techniques. It explores the use of models like Stochastic Gradient Descent (SGD) and k-Nearest Neighbors (k-NN) for accurate classification. The project also covers error analysis through confusion matrices and introduces advanced concepts such as multi-label classification.

## What I've Learned

- **Data Loading**: I learned how to download and load the MNIST dataset using `fetch_openml` from `sklearn`.
- **Data Preprocessing**: I implemented feature scaling using `StandardScaler` to improve model accuracy, from 85% to 90%.
- **Model Building**: I built and trained an SGD classifier, and later implemented a k-NN classifier for multi-label classification.
- **Error Analysis**: I performed an error analysis using confusion matrices to visually understand the classification errors.
- **Advanced Techniques**: I explored multi-label classification with k-NN, predicting both whether a digit is greater than or equal to 7 and whether it is odd.

## Key Features of the Project

- **SGD Classifier**: Uses a simple Stochastic Gradient Descent classifier to classify handwritten digits with a cross-validation accuracy of 85%, which increases to 90% after scaling.
- **Feature Scaling**: Implemented `StandardScaler` to scale the data and improve model performance.
- **Confusion Matrix Analysis**: Visualized classification errors using confusion matrices.
- **Multi-label Classification**: Explored k-Nearest Neighbors for predicting multiple labels at once, such as whether a digit is greater than or equal to 7 and whether it is odd.

## Getting Started

### Prerequisites

To run the project, you need to have the following Python libraries installed:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `joblib`
- `scipy`

You can install the necessary libraries by running:

```bash
pip install numpy matplotlib scikit-learn joblib scipy
```

### Running the Notebooks

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/mnist-classification.git
   cd mnist-classification
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter Notebooks:

   ```bash
   jupyter notebook MNIST_ML.ipynb
   jupyter notebook continuation_of_MNIST.ipynb
   ```

## Future Learning Goals

- **Model Optimization**: Experiment with additional models and hyperparameter tuning to improve classification accuracy.
- **Feature Engineering**: Develop new features and further optimize the dataset.
- **Deep Learning**: Explore neural network approaches, such as using a Convolutional Neural Network (CNN), to enhance classification performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
