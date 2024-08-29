# Heart Disease Prediction Using Genetic Algorithms and Ensemble Learning

## Project Overview

This project aims to enhance the prediction of heart disease using advanced machine learning techniques, specifically Genetic Algorithms (GA) for feature selection and Ensemble Learning for model improvement. The objective is to optimize model performance by selecting the most relevant features and employing ensemble methods for robust predictions.

## Dataset and Initial Setup

### Dataset Description

The dataset used for this project contains various medical attributes related to heart disease. It includes features such as cholesterol levels, resting blood pressure, and age. The primary target variable is the presence or absence of heart disease.

### Data Preparation

Initially, the dataset was preprocessed to standardize the feature values. Some features were dropped based on preliminary analysis and existing research to reduce complexity and improve model performance. This process involved:

- **Feature Dropping:** Features such as age, sex, cholesterol, resting blood pressure, and fasting blood sugar were removed based on their perceived relevance and redundancy.
- **Standardization:** The remaining features were standardized to ensure that all variables contributed equally to the model.

## Genetic Algorithm for Feature Selection

### Purpose of Genetic Algorithms

Genetic Algorithms (GAs) were employed to optimize feature selection. GAs are inspired by the process of natural selection and are used to identify the most relevant features by evolving a population of feature subsets over several generations.

### GA Implementation

- **Parameters:**
  - **Population Size:** 50 subsets of features.
  - **Number of Generations:** 100 iterations to evolve feature subsets.
  - **Crossover Rate:** 60% of the feature subsets were combined to create new subsets.
  - **Mutation Rate:** 3.3% chance to randomly alter features in a subset.
  - **Elite Count:** The top-performing subsets were directly carried over to the next generation.
  - **Stall Generation Limit:** The process stopped if there was no improvement for 10 consecutive generations.

The GA helped in identifying the optimal subset of features that maximize the model's performance. This subset was used to train and evaluate various machine learning models.

## Ensemble Learning and Model Evaluation

### Ensemble Learning Approach

To improve prediction accuracy and robustness, Ensemble Learning methods were applied. Ensemble methods combine multiple base models to produce a stronger overall model. Specifically:

- **Stacked Generalization (Stacking):** Multiple base models were trained, and their predictions were used as inputs for a meta-model. This meta-model (AdaBoostClassifier in this case) combined the predictions from the base models to make the final prediction.

### Base Models

Several base models were used in the ensemble, including:

- **Logistic Regression**
- **Gaussian Naive Bayes**
- **Decision Tree**
- **Support Vector Machine (SVM)**
- **Multi-layer Perceptron (MLP)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

These models were chosen for their diverse characteristics and abilities to capture different aspects of the data.

### Model Training and Evaluation

- **Cross-Validation:** K-Fold Cross-Validation was used to evaluate model performance. This technique involves dividing the dataset into K subsets and training the model K times, each time using a different subset as the validation set and the remaining subsets as training data.

- **Performance Metrics:** Key metrics included accuracy, sensitivity (recall), and specificity. These metrics were used to assess the effectiveness of the models in predicting heart disease.

## Conclusions

Conclusion and Comparison of Model Performance
This analysis evaluates the performance of various machine learning models using three different feature sets and validation methods. The results reveal critical insights into the effectiveness of the models and the impact of feature selection and validation techniques on their performance.

### 1. Using Article's Features
##### Holdout Method:
Looking at accuracy the articles Stacked-GA model performs better with 97.57% and ours with Stacked Model with 95.82%.
Their sensitivity is 96%, and specificity is 97% while ours is 95% for both.

In both sets of results, the stacked model significantly outperforms individual classifiers, highlighting the effectiveness of ensemble methods.

The article’s results are consistently higher across most models, which could be due to better hyperparameter tuning on their end or differences in the dataset used.

##### K-Fold Cross-Validation:
Their Stacked-GA model and our Stacked Model perform comparably across the holdout and different k-folds.

Their models generally show an increasing trend in performance as the number of k-folds increases, while our tends to stay consistent or drop as the number of folds increase.

### 2. Using Custom Features
##### Holdout Method:
Using our feature set the models shows improvements in the holdout across all the models for all sensitivity, specificity and accuracy.

##### K-Fold Cross-Validation:
But when we apply K-folds we see a sharp decrease in the performance of the models.

This maybe due to various reasons like overfitting or suboptimal hyperparameters, etc.

### 3. Overall Conclusion
In all cases, the Stacking Genetic Algorithm with AdaBoost performed head and shoulders above the rest of the algorithms even with GA.

#### Areas for Improvement

One key area for improvement is hyperparameter tuning such as trying out different estimators for AdaBoost, different depths for Random Forest or Decision Tree. Due to some technical limitations, we were unable to thoroughly explore and optimize hyperparameters for our models. Hyperparameter tuning is essential for squeezing out the best performance from our algorithms, and without it, we may not be reaching the full potential of our models. To address this, we plan to implement a systematic approach to hyperparameter optimization. This could involve using techniques like Grid Search and Random Search.

#### Next Steps

Looking ahead, one of our primary goals is to train our models on significantly larger datasets. Expanding the volume of data will help improve the model’s generalization capabilities and make it more robust. By incorporating additional data we can enhance the model's performance and reliability.

Another important step is to work with datasets that have a larger number of features as the one we used for this project has a small number of features. By integrating datasets that include a broader array of features, we aim to capture more nuanced information, which can lead to more accurate and insightful predictions. We plan to employ feature engineering techniques to create new features or select the most important ones, and consider dimensionality reduction methods such as Principal Component Analysis (PCA) to manage and streamline the data.

Additionally, we will focus on optimizing our code to speed up the training process which has been the primary bottleneck for this project. This includes exploring parallel processing options, utilizing GPU acceleration where possible, and improving data handling efficiency. By making these adjustments, we can reduce training time and accelerate our experimentation.

Lastly, a key enhancement for our project will involve expanding the number of models stacked together. Currently, our approach stacks 7 models, but increasing this to maybe 10 models will allow us to leverage a broader range of model predictions, which could improve overall performance. Additionally, we aim to implement a multi-level stacking strategy. Instead of the current 2-level approach, we could explore a 3 or 4-level stacking approach. This will enable us to build more complex metamodels that can integrate a richer set of predictions from different base models, potentially leading to more accurate and robust outcomes.

## References

Wang, Z., Zhang, Y., Chen, Z., Yang, H., Sun, Y., Kang, J., Yang, Y., & Liang, X. (2016). Application of ReliefF algorithm to selecting feature sets for classification of high resolution remote sensing image. https://doi.org/10.1109/igarss.2016.7729190

Yu, L., & Liu, H. (n.d.). Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution. https://cdn.aaai.org/ICML/2003/ICML03-111.pdf

Carr, J. (2014). An Introduction to Genetic Algorithms. https://www.whitman.edu/Documents/Academics/Mathematics/2014/carrjk.pd

Mienye, I. D., & Sun, Y. (2022). A Survey of Ensemble Learning: Concepts, Algorithms, Applications, and Prospects. IEEE Access, 10, 99129–99149. https://doi.org/10.1109/ACCESS.2022.3207287f
‌

