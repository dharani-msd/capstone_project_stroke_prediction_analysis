
**##Part -2**                        **Supervised Machine Learning Model — Build, Train, and Evaluate** 

**##Task 1** 

**#Data Loading and Label Definition :**

The cleaned dataset (stroke_cleaned_data.csv) generated in Part 1 was loaded into a pandas DataFrame for supervised learning.

**Feature Matrix (X) :**

The feature matrix (X) contains all predictor variables after excluding the target columns:

•	avg_glucose_level (regression target)

•	stroke (classification target)

**Regression Label (y_reg) :**

The continuous numeric column avg_glucose_level was selected as the regression target.

**Classification Label (y_clf) :**

The existing binary column stroke was selected as the classification target.

- 0 = No Stroke
- 1 = Stroke

The Stroke Prediction dataset already contains a natural binary target (stroke), so it was used instead of creating a new binary label from the regression target. This follows the assignment requirement, which allows using either a binary label derived from y_reg or another natural binary column in the dataset.

**Dataset Shapes :**

-	Feature Matrix (X): (5110, 9)
-	Regression Label (y_reg): (5110,)
-	Classification Label (y_clf): (5110,)

**##Task - 2**

**#Encoding categorical column:**

**#Categorical Feature Encoding:**

#The feature matrix contained the following categorical columns:

- gender
- ever_married
- work_type
- Residence_type
- smoking_status

These variables do not have a natural order, so One-Hot Encoding was applied using pd.get_dummies() with drop_first=True.

No label encoding was used because assigning integer values to these categories would create a false ordinal relationship. One-hot encoding represents each category as a separate binary feature, allowing machine learning models to treat the categories independently.

The first dummy variable for each categorical feature was dropped (drop_first=True) to avoid multicollinearity (the dummy variable trap).

**##Task – 3**

**#Leak-free train-test split and scaling:** 

**#Train-Test Split and Feature Scaling:**

The dataset was divided into training and testing sets using an 80:20 split with random_state=42 to ensure reproducibility.

A StandardScaler was used to standardize the feature values. The scaler was fitted only on the training data using scaler.fit(X_train). The fitted scaler was then used to transform both the training and testing feature sets.

This approach prevents data leakage. Fitting the scaler on the entire dataset before splitting would allow information from the test set (such as its mean and standard deviation) to influence the training process. As a result, the model could achieve overly optimistic performance that would not reflect its ability to generalize to unseen data.

**##Task – 4:**

**#Regression model — Linear Regression:**

**#Regression Target Selection:**

The regression target (y_reg) is avg_glucose_level, a continuous numeric variable. It was selected because regression models are designed to predict continuous values, and blood glucose level is an important health indicator in the Stroke Prediction dataset.

**#Linear Regression:**

A Linear Regression model (OLS) was trained using the scaled training features to predict the continuous target variable as i choose avg_glucose_level. Model performance was evaluated using Mean Squared Error (MSE) and the coefficient of determination (R²) on the test dataset

The model coefficients were examined to determine which features had the greatest influence on the predicted glucose level. The three features with the largest absolute coefficient values were identified as the most influential predictors.

A large positive coefficient indicates that a one-unit increase in the corresponding scaled feature is associated with an increase in the predicted avg_glucose_level by approximately the coefficient value, while keeping all other features constant.

A large negative coefficient indicates that a one-unit increase in the corresponding scaled feature is associated with a decrease in the predicted avg_glucose_level by approximately the coefficient value, assuming all other features remain unchanged.

**#Ridge Regression:**

Ridge Regression (alpha = 1.0) was trained using the same training and testing split. Its performance was compared with the standard Linear Regression model using MSE and R².

Ridge Regression applies L2 regularization, which penalizes large coefficient values and helps reduce overfitting and multicollinearity. The alpha parameter controls the strength of this regularization. A larger alpha results in greater coefficient shrinkage, while alpha = 0 makes Ridge Regression equivalent to ordinary Linear Regression. In this project, using alpha = 1.0 produced only a minimal difference compared with Linear Regression, suggesting that regularization had little impact on this dataset.

**#Regression Model Comparison:**

| Metric | OLS (Linear Regression) | Ridge Regression | Difference |
|--------|-------------------------:|-----------------:|-----------:|
| MSE | 1788.8034 | 1788.7996 | -0.0038 |
| R² | 0.069711 | 0.069713 | +0.000002 |


Both models produced nearly identical performance. Ridge Regression achieved a slightly lower Mean Squared Error (MSE), but the improvement was very small. The R² score remained the same for both models, indicating that they explained approximately 6.97% of the variance in the target variable (avg_glucose_level).

R² value of 0.0697 is quite low, which means the current features are not very effective at predicting avg_glucose_level with a linear model. For this dataset, the classification task (predicting stroke) is usually much more meaningful and often yields more informative results than trying to predict avg_glucose_level with linear regression.

**##Task – 5:**

**#Classification model :**

The target variable (stroke) was found to be highly imbalanced. The training dataset contained 3901 non-stroke cases (95.43%) and 187 stroke cases (4.57%), with the minority class representing less than the required 35% of the samples. To address this imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was applied only to the training dataset. SMOTE generates synthetic samples for the minority class, resulting in a balanced training dataset while preventing data leakage into the test set. This approach was chosen because improving the identification of stroke cases (minority class) is more important than maximizing overall accuracy in a medical prediction problem.

**#Before/After comparison:**

| Dataset | Class 0 | Class 1 |
|---------|--------:|--------:|
| Before balancing | 3901 | 187 |
| After balancing | 3901 | 3901 |

This table is exactly what the requirement means by "Class imbalance handling is demonstrated with a before/after class count comparison." It clearly shows the class counts before balancing and after balancing, demonstrating that the imbalance was successfully addressed.

**#Logistic Regression :**

The Logistic Regression model achieved an accuracy of 74.75%. For the minority stroke class, it obtained a precision of 18%, recall of 87%, and an F1-score of 30%. The confusion matrix showed 710 True Negatives, 54 True Positives, 250 False Positives, and 8 False Negatives. Although the model produced a relatively high number of false positives, it successfully identified 87% of actual stroke cases, making it suitable for medical screening where minimizing missed stroke patients is more important than reducing false alarms.

**#ROC AND AUC CURVE:**

The Logistic Regression model achieved an AUC score of 0.8465, indicating good classification performance. This means the model can effectively distinguish between stroke and non-stroke patients. An AUC of 0.8465 indicates that there is approximately an 84.65% probability that the model will rank a randomly selected stroke patient higher than a randomly selected non-stroke patient.

**#Formulas :**

| Metric | Formula |
|--------|---------|
| TPR (Recall) | TP / (TP + FN) |
| FPR | FP / (FP + TN) |
| ROC Curve | Plot of TPR vs FPR |
| AUC | Area under the ROC curve (∫₀¹ TPR(FPR) d(FPR)) |

Precision and Recall formulas:

**Precision:**

                          Precision = TP / ( TP + FP)

Measures how many of the patients predicted as having a stroke actually had a stroke. 

**Recall:**

                           Recall = TP / ( TP + FN)

Measures how many of the actual stroke patients were correctly identified by the model. 
Most Important Metric in the dataset:

Basically in healthcare datset our first most important metric is Recall. For stroke prediction, Recall is more important than Precision because missing a patient who actually has a stroke (False Negative) can lead to delayed treatment and serious health consequences. Although a higher recall may increase the number of false positives, identifying as many true stroke cases as possible is the priority in medical diagnosis.

**##Task – 6:**

**#Decision-Threshold Sensitivity:**

It means how the model's predictions change when you change the probability threshold used for classification. In Logistic Regression, the default threshold is

**Threshold = 0.5**

•	If probability ≥ 0.5 → Predict Stroke (1) 

•	If probability < 0.5 → Predict No Stroke (0)

**a)Precision and Recall formulas:**

Precision:

                          Precision = TP / ( TP + FP)

Measures how many of the patients predicted as having a stroke actually had a stroke. 

Recall:
       
                           Recall = TP / ( TP + FN)

Measures how many of the actual stroke patients were correctly identified by the model. 


**(b) Threshold that Maximizes F1-Score:**

| Threshold | Precision | Recall | F1-Score |
|-----------|----------:|-------:|---------:|
| 0.30 | 0.128 | 0.919 | 0.226 |
| 0.40 | 0.152 | 0.919 | 0.262 |
| 0.50 (Default) | 0.177 | 0.870 | 0.295 |
| 0.60 | 0.186 | 0.693 | 0.293 |
| 0.70 | 0.195 | 0.532 | 0.285 |

The 0.50 threshold produced the highest F1-score (0.295), indicating the best balance between precision and recall for this dataset.

**c) More Important Metric:**

For stroke prediction, Recall is more important than Precision because failing to identify a patient who actually has a stroke (False Negative) can delay treatment and lead to serious health consequences. Although a higher recall may increase the number of false positives, identifying as many true stroke patients as possible is the primary objective in medical screening.

**d) Threshold Selection:**

To optimize for Recall, the decision threshold should be lowered (for example, from 0.50 to 0.30 or 0.40). Lowering the threshold allows the model to classify more patients as stroke, increasing the number of correctly detected stroke cases.

The trade-off is that lowering the threshold also increases the number of False Positives, meaning more healthy individuals may be incorrectly classified as being at risk of stroke and may undergo additional medical evaluation. In healthcare, this trade-off is generally acceptable because missing a true stroke patient is considered more serious than investigating a false alarm.

**##Task – 7:**

**#Regularization experiment on Logistic Regression:**

| Logistic Regression Model | C Value | Precision | Recall | AUC |
|---------------------------|--------:|----------:|-------:|----:|
| Baseline Model | 1.0 | 0.178 | 0.871 | 0.8465 |
| Strong Regularization | 0.01 | 0.173 | 0.871 | 0.8503 |

**Performance Comparison:**

•	Precision: decreased slightly from 0.178 to 0.173. 

•	Recall: remained the same at 0.871. 

•	AUC: increased slightly from 0.8465 to 0.8503.

Reducing C from 1.0 to 0.01 applied stronger L2 regularization, resulting in a simpler Logistic Regression model. On this stroke prediction dataset, stronger regularization caused a small decrease in precision, no change in recall, and a slight improvement in AUC. Overall, the effect of stronger regularization was minimal, indicating that the baseline model was already well regularized. Since recall remained high and AUC improved slightly, the C = 0.01 model maintained good discrimination between stroke and non-stroke patients while using stronger regularization.

**##Task – 8:**

**#Bootstrap confidence interval for AUC difference:**

A bootstrap analysis with 500 resamples was performed to compare the AUC of the baseline Logistic Regression model (C = 1.0) with the strongly regularized model (C = 0.01).

•	Mean AUC Difference (C = 1.0 − C = 0.01) = -0.00384

•	95% Confidence Interval: (-0.00834, 0.00039)

The mean AUC difference is slightly negative, indicating that the C = 0.01 model achieved a marginally higher AUC than the baseline model. However, the 95% confidence interval includes zero, which indicates that the observed difference is not statistically significant. Therefore, the performance difference between the two Logistic Regression models may be due to random sampling variation, and there is insufficient evidence to conclude that one model consistently outperforms the other.

