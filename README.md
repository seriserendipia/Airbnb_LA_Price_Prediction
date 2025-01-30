# Analysis of Los Angeles Airbnb Listing Prices

## Objective
The goal of this analysis is to explore the factors influencing Airbnb listing prices in Los Angeles using the most recent data downloaded from Inside Airbnb.

---

## Data Preprocessing

### Categorical Variables
- **Challenge**: Variables such as neighborhoods and property types contained too many small categories.
- **Solution**:
  - For columns such as `property_type`. Consolidated small categories using domain knowledge or retained only the top N categories by frequency.Then one-hot encode the categorical data.
  - Applied frequency encoding to improve model performance, especially for the `neighborhood_cleansed` variable.

### Text Features
- **Analysis**: Extracted basic text attributes such as word count and frequency from columns like `description`.
- **Hypothesis**: Listings with higher word counts in their descriptions tend to be of higher quality and have higher prices.

### Numerical Variables
- **Normalization**: Standardized numerical columns to ensure uniformity across features.
- **Missing Values**: Filled missing values with column means to maintain dataset integrity.
- **Target Variable (`price`)**:
  - Observed high skewness in the price distribution using histograms.
  - Applied log transformation to normalize the target variable, which improved model performance.

---
## Feature Selection Approach
| Feature Set              | MSE        | R-squared  |
|--------------------------|------------|------------|
| Full Dataset             | 0.3038     | 0.7261     |
| Top 10 Features          | 0.3721     | 0.6644     |
| Manual Selected Features | 0.4265     | 0.6154     |

- **Feature Importance**:
Although the full model outperformed the reduced model (feature importance top 10 features), we opted for the top 10 feature approach due to its computational efficiency and interpretability. However, this decision acknowledges a slight compromise in predictive performance.


- **Manual Feature Selection**:
In addition to automated methods such as feature importance and correlation analysis, domain knowledge and common sense were applied to manually select features that seemed intuitively important for predicting Airbnb prices. However, this approach did not result in better performance compared to models relying on algorithmically selected features.



---

## Models Evaluation
| Model         | R²    | MSE    | RMSE   |
|---------------|--------|--------|--------|
| ElasticNet    | 0.5630 | 0.4846 | 0.6961 |
| RandomForest  | 0.7257 | 0.3042 | 0.5515 |
| DecisionTree  | 0.4694 | 0.5884 | 0.7671 |
| SVR           | 0.6842 | 0.3502 | 0.5918 |
| CatBoost      | 0.7284 | 0.3011 | 0.5488 |

The best model is CatBoost

### CatBoost Hyperparameter Tuning

Best Hyperparameters
- **Iterations**: 400  
- **L2 Leaf Regularization**: 3  
- **Learning Rate**: 0.3

Performance Metrics
- **Root Mean Squared Error (RMSE)**: 0.4803  
- **R-squared (R²)**: 0.7568  


## Catboost best model:  5-Fold Cross Validation Results

| Fold | RMSE       | R-squared   |
|------|------------|-------------|
| 1    | 0.3744     | 0.8119      |
| 2    | 0.3867     | 0.8083      |
| 3    | 0.3808     | 0.8170      |
| 4    | 0.3831     | 0.8096      |
| 5    | 0.3974     | 0.8025      |

### Summary
- **Mean RMSE**: 0.3845  
- **Mean R-squared**: 0.8099




---

## Price Prediction Using the Best Model

Using the best-performing model (CatBoost Regressor), the predicted price for a 3-bedroom, 2-bathroom, instant bookable, Entire home/apt listing that accommodates 5 is:

### Predicted Price:
- **$1157.05**
---

## Findings

### 1. Neighborhoods and Locations
- Frequency encoding of the `neighborhood_cleansed` variable outperformed other encoding methods.

### 2. Property Type and Features
- Entire homes and luxury property types consistently command higher prices compared to shared or private rooms.
- Key features such as the number of bedrooms, bathrooms, and amenities positively correlated with price.

### 3. Textual Insights
- Listings with detailed descriptions (higher word counts) and high counts of amenities tend to have higher prices.
- Text analysis adds significant value in predicting perceived listing quality.

### 4. Price Distribution
- The original price distribution was heavily skewed, with a long tail of high-priced listings.
- Log transformation reduced skewness and improved the predictive power of the models.


# Questions:
1. When I applied log transformation to y, should I use y_logged to calculate RMSE and R2 or should I use the original y to calculate?
2. Can I simply delete any rows that contains NaN if the model performs very bad with NaNs? And should this procedure happens before or after train/test set splitting
