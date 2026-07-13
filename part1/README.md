Stroke Prediction using Machine Learning and LLM

**##Part 1 - STROKE PREDICTION DATA ANALYSIS**

**#Project Overview**

I selected the Stroke Prediction Dataset because it is a publicly available healthcare dataset that is suitable for a machine learning classification project. The dataset contains more than 5,000 records and includes both numeric and categorical features, making it appropriate for data preprocessing and analysis. It also contains missing values, which provide an opportunity to perform data cleaning techniques such as missing value imputation and data type conversion.

This dataset aligns well with the Data-to-AI Lifecycle because it allows the complete workflow to be demonstrated, including data collection, data inspection, data cleaning, exploratory data analysis (EDA), feature engineering, model training, and model evaluation. In addition, the problem of predicting stroke risk has practical importance in healthcare, where early prediction can support timely medical intervention and improve patient outcomes.

**#Dataset**

This project uses the Stroke Prediction Dataset obtained from Kaggle. The dataset contains patient demographic information, medical history, lifestyle factors, and a binary target variable (stroke) indicating whether a patient has experienced a stroke.

The dataset was downloaded from Kaggle, saved locally, and loaded into a pandas DataFrame using the read_csv() function. It was then cleaned and preprocessed before performing exploratory data analysis (EDA), machine learning model development, and LLM integration.

**#Dataset Summary**

- Rows: 5,110
- Columns: 12
- Target Variable: stroke

**#Development Environment / Tools & Technologies:**

- Python 3
- Google Colab

**#Libraries**

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Requests
- JSONSchema
- OpenAI

**##Task – 1:**

**#Loading the Dataset:**

The stroke prediction dataset was loaded into a pandas DataFrame using the read_csv() function. This dataset served as the foundation for all subsequent data cleaning, exploratory data analysis (EDA), visualization, and machine learning tasks performed in the project.

""" import pandas as pd

df = pd.read_csv("healthcare_dataset_stroke_data.csv") """

**#Delete columns (not relationship)**

The id column is just a unique number assigned to each patient. It has no relationship to whether someone has a stroke, so it doesn't help the model learn.

**##Task – 2:**

**#Null value analysis (Handling Missing Values (bmi column)) :**

The bmi column contained missing values and it is around 3.9% of missing values are there. These missing values were replaced using the median of the column instead of the mean. The median is less affected by extreme values and skewed distributions, making it a more robust measure of central tendency. Therefore, using the median helps preserve the overall distribution of the data and reduces the influence of outliers. I found null values in dataset for using df.isnull().sum() and (df.isnull().sum() / df.shape[0]) * 100) . fill numeric columns with the column median using fillna(df[col].median()).

**##Task -3:**

**#Duplicate detection and removal:**

I checked the dataset for duplicate records using df.duplicate().sum(). The result was 0, which means there were no duplicate rows, so no records needed to be removed

**##Task – 4:**

**#Data Type Correction:**

Data type correction was performed to ensure that each column had the appropriate data type before analysis and model development. Numeric columns were converted to numeric data types, while categorical columns were retained as object or category types where appropriate.

The pandas to_numeric() function was used to convert values to numeric format, with invalid values handled using errors="coerce".

"""df[column] = pd.to_numeric(df[column], errors="coerce")"""

Correcting data types improves data quality, enables accurate statistical analysis, and ensures compatibility with machine learning algorithms.

In this data has already in correct datatype so I just convert few repeated values columns into categorical datatype column why because storing them as category instead of object reduces memory usage and can improve the performance of data processing and machine learning algorithms. Memory usage before and after the conversion using df.memory_usage(deep=True).sum()

In this data Memory before conversion: 1656573 bytes and Memory after conversion: 272532 bytes.

**##Task -5:**

**#Descriptive statistics and skewness:**

df.describe() is a quick health check of the numeric data. It summarizes the dataset and helps identify missing values, the range of values, and the overall distribution before data cleaning.

**#Skewness Analysis:**

Skewness was calculated using (df[col].skew()) to understand the distribution of the continuous numeric variables in the dataset.

• Age: The skewness value is -0.137, which is close to zero. This indicates that the age column is approximately symmetric with a slight negative (left) skew. The data is fairly evenly distributed.

• Average Glucose Level: The skewness value is 1.572, indicating a highly positive (right) skewed distribution. This means that most patients have lower glucose levels, while a small number of patients have very high glucose levels.

• BMI: The skewness value is 1.088, indicating a positively (right) skewed distribution. Most patients have BMI values around the lower to middle range, while a few patients have much higher BMI values. Because the BMI distribution is skewed, the median was used to replace missing values instead of the mean, as the median is less affected by extreme values and outliers

**##Task – 6:**

**#Outlier Detection Using the IQR Method:**

Outliers were identified using the Interquartile Range (IQR) method. The IQR measures the spread of the middle 50% of the data and is calculated as:

IQR = Q3 − Q1

where: • Q1 = First Quartile (25th percentile) • Q3 = Third Quartile (75th percentile) A data point is considered an outlier if it satisfies either of the following conditions:

• Lower Bound = Q1 − 1.5 × IQR

• Upper Bound = Q3 + 1.5 × IQR

Any values below the lower bound or above the upper bound were identified as potential outliers. The IQR method was chosen because it is robust to skewed distributions and is not affected by extreme values.

The analysis showed that the Age column did not contain any outliers. The Average Glucose Level column contained 627 outliers, indicating that some patients have unusually high glucose levels. The BMI column contained 126 outliers, representing patients with unusually high or low body mass index values.

The outliers were not removed because they may represent genuine patient measurements rather than data entry errors. Removing them could result in the loss of important medical information. Therefore, the outliers were retained and will be considered during model development in Part 2.

**##Task – 7:**

**#Visualizations:**

• A Line Plot : It was created using the avg_glucose_level column. The x-axis represents the patient index, and the y-axis represents the average glucose level. The plot shows how glucose levels vary across different patients and helps identify patterns and unusually high values.

• A Bar Chart : A bar chart comparing the mean of one numeric column across categories of one categorical column .The x-axis represents categorical column (e.g., gender, work_type, smoking_status) and y-axis represents the mean of a numeric column (e.g., avg_glucose_level, bmi, age)

• A Histogram : The avg_glucose_level column was created using sns.histplot() with 20 bins. This column was identified as the most skewed numeric feature in the dataset (skewness = 1.57). The x-axis represents Average level glucose and y-axis represents Frequency.The distribution is highly right-skewed (positively skewed), with most observations concentrated between lower and moderate glucose levels. A small number of observations have much higher glucose values, creating a long tail on the right side of the histogram. This indicates that the data is not normally distributed and contains a few high-value observations (potential outliers).

• A Scatter Plot : Interpretation of (age vs avg_glucose_level) A scatter plot was created to examine the relationship between age and average glucose level. The plot shows a weak positive relationship, indicating that average glucose levels tend to increase slightly as age increases. However, the points are widely scattered, suggesting that the relationship is not strong and that other factors may also influence glucose levels.

• A Box Plot : Interpretation of Average Glucose Level by Stroke Status The box plot compares the average glucose levels of patients with and without stroke. The stroke = 1 group has a higher median average glucose level than the stroke = 0 group, indicating that patients who experienced a stroke generally have higher glucose levels. The stroke = 1 group also shows a wider spread of values, suggesting greater variability in glucose levels. Outliers are present in both groups, representing patients with unusually high glucose levels.

**##Task – 8:**

**#Correlation Heatmap:**

A correlation heatmap was created using the numeric columns in the dataset. The strongest absolute correlation was observed between age and BMI. The highest absolute correlation in the dataset was 0.32 between age and BMI. This indicates a weak positive relationship, meaning that BMI tends to increase slightly as age increases. Although this is the strongest correlation among the numeric variables, it is still relatively weak.

This correlation does not imply that age directly causes changes in BMI. Other factors such as diet, physical activity, genetics, and overall health could influence both age-related health and BMI, making them plausible alternative explanations.

The two numeric columns with the highest absolute skewness were avg_glucose_level (1.57) and bmi (1.09). For each column, both the mean and median were calculated before imputation. Since both columns are positively skewed, the median was chosen to fill missing values. In positively skewed distributions, extreme high values pull the mean upward, making the median a more representative measure of central tendency. After applying median imputation using fillna(), isnull().sum() confirmed that no missing values remained in either column.

**##Task – 9a:**

**#Imputation strategy comparison:**

The two numeric columns with the highest absolute skewness were avg_glucose_level (1.57) and bmi (1.09). For each column, both the mean and median were calculated before imputation. Since both columns are positively skewed, the median was chosen to fill missing values. In positively skewed distributions, extreme high values pull the mean upward, making the median a more representative measure of central tendency. After applying median imputation using fillna(), isnull().sum() confirmed that no missing values remained in either column.

**##Task – 9b:**

**##Spearman rank correlation :**

#Pearson Correlation = Straight Line and Linear

It measures whether two variables change together in a straight-line pattern.

#Spearman Correlation = Ranks + Monotonic relationship

It converts values into ranks and checks whether the variables generally move in the same direction, even if the relationship is curved.

In these correlation interpretation we just find top three largest difference pairs of correlation

**1)Age and Average Glucose Level**

-Pearson = 0.238

-Spearman = 0.145

Result = Pearson > Spearman

The age and average glucose level pair has a higher Pearson correlation than Spearman correlation. This suggests that the relationship is approximately linear rather than monotonic but non-linear. Therefore, Pearson correlation is the preferred measure for feature-selection guidance for this pair.

**2)Hypertension and Average Glucose Level**

-Pearson = 0.174

-Spearman = 0.114 

Again, Result = Pearson > Spearman

The hypertension and average glucose level pair shows a slightly higher Pearson correlation than Spearman correlation, indicating an approximately linear but weak relationship. Pearson correlation is more suitable for feature selection because it better represents the observed association.

**3)Heart Disease and Average Glucose Level**

-Pearson = 0.162

-Spearman = 0.106

Again, Result = Pearson > Spearman

The heart disease and average glucose level pair also has a higher Pearson correlation than Spearman correlation. This suggests an approximately linear relationship, although the association is weak. Pearson correlation is therefore preferred for feature-selection guidance. The relationships are approximately linear, and Pearson correlation is the more appropriate measure for feature selection for these variable pairs.

**##Task – 9c:**

**#Grouped Aggregation:**

The grouped aggregation of average glucose level by work type showed that the Self-employed group had the highest mean average glucose level (112.65) and also the highest standard deviation (51.72). The high standard deviation indicates considerable variation in glucose levels within the Self-employed group, suggesting that work type alone is insufficient to reliably predict the target outcome. The ratio of the highest group mean to the lowest group mean is 1.19 (112.65 / 94.40). This relatively small difference suggests that while work type may provide some predictive signal, it is unlikely to be a strong standalone predictor and should be used together with other features in the predictive model.It is find to compute with df.groupby(categorical_col)[numeric_col].agg(['mean', 'std', 'count'])

**##Task – 10:**

**#Save the Cleaned Dataset:**

The cleaned dataset was saved as cleaned_data.csv by using df.to_csv('stroke_cleaned_data.csv', index=False) after all preprocessing steps were completed. The file was saved without the DataFrame index and will be used as the input dataset for Parts 2 and 3 of the project.




