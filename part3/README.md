**##PART -3**  

                      **Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline**
                                         
**##TASK – 1:**

**##Decision Tree baseline:**

The baseline Decision Tree (max_depth=None) achieved a training accuracy of 99.98% and a test accuracy of 90.41%.

The model shows signs of overfitting because the training accuracy is nearly perfect, while the test accuracy is significantly lower. This indicates that the model has learned the training data, including noise and dataset-specific patterns, instead of learning only the underlying relationships.

Decision Trees are considered high-variance models because they build the tree greedily by selecting the best split at each node without revisiting earlier decisions. When there are no constraints on the tree depth (max_depth=None), the tree continues splitting until it fits the training data almost perfectly. As a result, even small changes in the training data can produce a different tree structure, leading to reduced generalization performance on unseen data.

**##TASK – 2:**

**##Controlled Decision Tree:**

A second Decision Tree was trained using max_depth=5 and min_samples_split=20.
The controlled model achieved a training accuracy of 95.47% and a test accuracy of 93.93%.
The max_depth parameter limits how deep the tree can grow. Restricting the depth reduces the model's variance and helps prevent it from memorizing the training data, although it may introduce a small amount of bias.

The min_samples_split parameter requires a node to contain at least 20 samples before it can be split. This prevents the tree from creating branches based on very small subsets of the data, reducing the likelihood of fitting noise.

Compared with the unconstrained Decision Tree, the controlled tree has a much smaller gap between the training and test accuracies (approximately 1.54% versus 9.57%). This indicates that the controlled tree generalizes better to unseen data and exhibits substantially less overfitting while maintaining strong predictive performance.

**Constrained vs. Unconstrained Decision Tree:**

•	The unconstrained Decision Tree achieved very high performance on the training data but lower performance on the test data, resulting in a large train/test gap. This indicates that the model overfitted the training data.

•	The constrained Decision Tree (max_depth=5) had a smaller train/test gap, showing that it generalized better to unseen data. Limiting the tree depth helped reduce overfitting and produced a more reliable model.

**##Task – 3:**

**##Gini vs Entropy Comparison:**

Two Decision Tree models were trained using max_depth=5, one with the Gini criterion and the other with the Entropy criterion.

•	Gini Test Accuracy: 93.74%

•	Entropy Test Accuracy: 93.93%

The Gini impurity is calculated as:

**Gini = 1 − Σ(pi²)**

where pi is the probability of each class in a node.

The Entropy is calculated as:

**Entropy = −Σ(pi × log₂(pi))**

where pi is the probability of each class.

A node with Gini = 0 is called a pure node, meaning that all samples in the node belong to a single class. Since the node contains only one class, no further splitting is required.

In this experiment, both criteria produced very similar results. The Entropy criterion achieved a slightly higher test accuracy (93.93%) than the Gini criterion (93.74%), with a difference of only 0.19%. This indicates that both impurity measures are effective for this dataset, although Entropy performed marginally better.

**##Task – 4:**

**##Classification:**

A Random Forest classifier was trained using 100 decision trees (n_estimators=100) with a maximum tree depth of 10 (max_depth=10).

**Results**

•	Training Accuracy: 96.55%

•	Test Accuracy: 93.93%

•	ROC-AUC: 0.8244

The small difference between the training and test accuracies (approximately 2.62%) indicates that the Random Forest generalizes well to unseen data and exhibits much less overfitting than a single unconstrained Decision Tree.

**Top 5 Most Important Features:**

**Rank	Feature	Importance**

1	Age	0.3433

2	Bmi	0.3246

3	Residence_type_Urban	0.0439

4	heart_disease	0.0418

5	Hypertension	0.0412

**Feature Importance**

Random Forest computes feature importance by measuring the average reduction in Gini impurity produced by each feature across all decision trees in the forest. Every time a feature is used to split a node, it reduces the impurity of that node. These reductions are accumulated over all splits and all trees, and then normalized to produce an overall importance score. Features with larger importance values contribute more to the model's predictions.

This differs from a Linear Regression coefficient, which represents the expected change in the predicted value for a one-unit increase in a feature while keeping all other features constant. Feature importance in Random Forest reflects how useful a feature is for making decision-tree splits, whereas a linear regression coefficient represents the direction and magnitude of a linear relationship.

**Bagging Concept**

Random Forest is an ensemble learning algorithm that uses bagging (Bootstrap Aggregating). Each decision tree is trained on a bootstrap sample, which is created by randomly sampling the training data with replacement. As a result, each tree sees a slightly different version of the training dataset. In addition, at every split, the algorithm considers only a random subset of √(number of features) rather than all available features. These two sources of randomness make the individual trees less correlated. The final prediction is obtained by combining the predictions from all trees (majority voting for classification). This ensemble averaging reduces variance, improves generalization, and produces a more stable model than a single deep Decision Tree.


**##Task – 4a:**

**##Feature Ablation Study:**

The five features with the lowest Random Forest feature importance scores were removed, and a second Random Forest model was trained using the same hyperparameters as the original model.

**Results**

•	Full Model ROC-AUC     :   0.8244

•	Reduced Model ROC-AUC  :   0.8269

The reduced model achieved a slightly higher ROC-AUC than the full model. This suggests that the five removed features were largely uninformative and may have introduced a small amount of noise into the model. Removing these features did not degrade predictive performance and instead produced a small improvement in the model's ability to distinguish between the two classes.

From a deployment perspective, using a lower-dimensional model can reduce inference time, simplify data collection, and lower maintenance costs. Such a simplification is beneficial when the reduction in model complexity does not cause a meaningful decrease in predictive performance. In this experiment, the reduced model maintained—and slightly improved—ROC-AUC, indicating that removing the least important features is an acceptable trade-off for a simpler production model.

**##Task – 5:**

**##Cross-validated comparison:**

Five-fold Stratified Cross-Validation with ROC-AUC scoring was used to compare the performance of four classification models.

| Model | Mean ROC-AUC | Standard Deviation |
|-----------------------------|-------------:|-------------------:|
| Logistic Regression | 0.8316 | 0.0107 |
| Gradient Boosting | 0.8158 | 0.0087 |
| Random Forest | 0.7738 | 0.0136 |
| Decision Tree (max_depth=5) | 0.7566 | 0.0327 |

Among the evaluated models, Logistic Regression achieved the highest mean ROC-AUC (0.8316), indicating the best overall predictive performance. Gradient Boosting achieved the lowest standard deviation (0.0087), showing the most consistent performance across the five folds. The Decision Tree produced both the lowest mean ROC-AUC and the highest standard deviation, indicating lower predictive performance and greater sensitivity to variations in the training data.

Cross-validation provides a more reliable estimate of generalization performance than a single train-test split because the model is trained and evaluated on multiple different partitions of the training data. Averaging the performance across several folds reduces the influence of any single data split and provides a more robust assessment of how well the model is likely to perform on unseen data.

**##Task – 6:**

**##Hyperparameter Tuning with GridSearchCV:**

A Pipeline consisting of SimpleImputer, StandardScaler, and RandomForestClassifier was used with GridSearchCV to identify the best hyperparameter combination. The search space included:

•	n_estimators: 50, 100, 200

•	max_depth: 5, 10, None

•	min_samples_leaf: 1, 5

This produced 18 hyperparameter combinations (3 × 3 × 2). Using 5-fold Stratified Cross-Validation, each combination was evaluated on five folds, resulting in a total of 90 model training runs.

The best-performing hyperparameters were:

•	n_estimators: 200

•	max_depth: 5

•	min_samples_leaf: 5

The corresponding mean cross-validated ROC-AUC was 0.8237.

The selected parameters indicate that a moderately sized forest with shallow trees and larger leaf nodes provided the best balance between predictive performance and generalization. Limiting the tree depth and requiring at least five samples per leaf helps reduce overfitting, while increasing the number of trees improves the stability of the ensemble.

**##Grid Search vs. Randomized Search**

Grid Search evaluates every possible combination of the specified hyperparameters, ensuring that the best combination within the predefined search space is found. However, this exhaustive approach becomes computationally expensive as the number of hyperparameters and candidate values increases. Randomized Search evaluates only a random subset of the possible combinations, making it much faster and more suitable for large search spaces. Although Randomized Search may not always identify the absolute best combination, it often finds a near-optimal solution with substantially lower computational cost.

**##Task – 7:**

**##Manual Learning Curve Analysis:**

The tuned Random Forest model was trained using 20%, 40%, 60%, 80%, and 100% of the training data.

1.	Training AUC: The training AUC gradually decreased from 0.9218 to 0.8832 as the training set grew. This indicates that the model overfits less when trained on larger datasets and generalizes better. 

2.	Test AUC: The test AUC generally increased from 0.8230 to 0.8574 as more training data was used. Although there was a slight fluctuation at 40%, the overall trend shows improved performance with additional training data. 

3.	Conclusion: Since the test AUC was still increasing at the 100% training fraction, the model appears to be data-limited rather than model-capacity-limited. This suggests that collecting more training data could further improve predictive performance.

| Training Fraction | Training AUC | Test AUC |
|------------------:|-------------:|---------:|
| 20% | 0.9218 | 0.8230 |
| 40% | 0.9107 | 0.8220 |
| 60% | 0.8979 | 0.8486 |
| 80% | 0.8855 | 0.8558 |
| 100% | 0.8832 | 0.8574 |

**##Task – 8:**

**##Serialize the best model:**

The best-performing model, Tuned Random Forest, was saved to a file named best_model.pkl using the joblib library. The saved model was then loaded back into the notebook using joblib.load(), and predictions were successfully made on two hand-crafted test samples. This confirms that the serialized model can be reloaded and used for future predictions without retraining.

**##Task – 9:**

**##Summary comparison table:**

| Model | 5-Fold CV Mean ROC-AUC | 5-Fold CV Std ROC-AUC | Test ROC-AUC |
|--------------------------------|----------------------:|----------------------:|-------------:|
| Logistic Regression | 0.8316 | 0.0107 | 0.8465 |
| Decision Tree (max_depth=5) | 0.7566 | 0.0327 | 0.5718 |
| Random Forest | 0.7738 | 0.0136 | 0.8244 |
| Gradient Boosting | 0.8158 | 0.0087 | 0.8390 |
| Tuned Random Forest | **0.8237** | **0.0111** | **0.8574** |

**Recommended Model**

I recommend the Tuned Random Forest because it achieved the highest test ROC-AUC (0.8574) among all the models. Hyperparameter tuning improved its performance compared to the baseline Random Forest (from 0.8244 to 0.8574). This model performed the best on unseen data, making it the most suitable choice for stroke prediction.
