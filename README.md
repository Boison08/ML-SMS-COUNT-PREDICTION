# SMS Count Prediction - Machine Learning Project

## Overview
This project implements a **Linear Regression model** to predict SMS message counts based on various features. The model analyzes the relationship between different variables and the total SMS count, providing insights into communication patterns.

## Author
**Simeon Boison**

## Project Structure
```
.
├── Simeon Boison.ipynb          # Main Jupyter Notebook with complete analysis
├── data.zip                     # Compressed dataset for training and testing 
├── Model Analysis.txt           # Model analysis results and insights
├── README.md                    # This file
└── .gitignore                   # Git ignore file
```

## Features
- **Data Loading & Exploration**: Load and analyze SMS data
- **Data Preprocessing**: Handle missing values
- **Exploratory Data Analysis (EDA)**: 
  - Correlation matrix and heatmap visualization
  - Statistical summary of data
- **Model Development**: Linear Regression implementation
- **Model Evaluation**: Comprehensive accuracy metrics
- **Predictions**: Generate predictions on training and test data
- **Visualization**: Scatter plots for predicted vs actual values

## Dataset
- **Source**: `data.csv`
- **Target Variable**: `total_sms` (SMS message count)
- **Features**: Multiple independent variables (excluding datetime)
- **Data Split**: 67% training, 33% testing

## Model Performance
The Linear Regression model achieves:
- **Training Accuracy (R² Score)**: 78.64%
- **Testing Accuracy (R² Score)**: 77.92%
- **Mean Absolute Error (Train)**: 2.2955
- **Mean Absolute Error (Test)**: 2.2830
- **Root Mean Squared Error (Test)**: 8.0241

### Interpretation
- The model explains approximately **78% of the variance** in SMS counts
- Minimal overfitting observed (training and test scores are similar)
- Strong predictive performance with consistent metrics across train/test sets

## Evaluation Metrics Used
1. **R² Score**: Coefficient of determination (0-1, higher is better)
2. **Mean Absolute Error (MAE)**: Average absolute prediction error
3. **Mean Squared Error (MSE)**: Penalizes larger errors
4. **Root Mean Squared Error (RMSE)**: MSE in original units
5. **Mean Absolute Percentage Error (MAPE)**: Percentage-based error metric

## Technologies & Libraries
- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **scikit-learn**: Machine learning models and metrics
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Project
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Simeon Boison.ipynb"
   ```

2. Ensure `data.csv` is in the same directory as the notebook

3. Run all cells sequentially to:
   - Load and explore the data
   - Train the Linear Regression model
   - Evaluate model accuracy
   - Generate predictions and visualizations

## Key Steps in the Analysis

### 1. Data Preparation
- Load CSV data using pandas
- Remove rows with missing datetime values
- Generate correlation matrix

### 2. Feature Engineering
- Separate independent variables (X) and dependent variable (Y)
- Exclude non-numeric columns (datetime)

### 3. Model Training
- Split data: 67% training, 33% testing (random_state=42)
- Initialize and train Linear Regression model
- Generate predictions on both datasets

### 4. Model Evaluation
- Calculate R² score for both train and test sets
- Compute MAE, MSE, RMSE, and MAPE
- Compare training vs test performance

### 5. Visualization
- Correlation heatmap
- Predicted vs Actual scatter plots
- Regression line overlay

## Results & Insights
- The model demonstrates strong predictive capability
- No significant overfitting detected
- Model generalizes well to unseen test data
- Consistent performance across evaluation metrics

## Files Description
- **Simeon Boison.ipynb**: Contains all code, visualizations, and outputs
- **data.csv**: Raw dataset with SMS counts and features
- **numbers_and_squares.csv**: Supporting numerical dataset
- **Model Analysis.txt**: Detailed analysis notes

## Future Improvements
- Explore additional regression algorithms (Ridge, Lasso, Random Forest)
- Implement hyperparameter tuning
- Add cross-validation for more robust evaluation
- Perform feature selection and engineering
- Implement ensemble methods for better predictions

## License
MIT License






