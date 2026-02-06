# ==================== COMPLETE INSURANCE EDA + PREDICTION + EXCEL EXPORT ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import openpyxl
import os

print("🔍 STARTING COMPLETE INSURANCE ANALYSIS...")

# ==================== 1. LOAD YOUR DATA ====================
try:
    df = pd.read_csv(r'C:\Users\jainp\Downloads\insurance (1).csv')
    print("✅ Data loaded successfully!")
    print(f"📁 Dataset Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    print("Please check the file path and try again!")
    exit()

# ==================== 2. EDA - EXPLORATORY DATA ANALYSIS ====================
print("\n" + "="*50)
print("📊 EXPLORATORY DATA ANALYSIS (EDA)")
print("="*50)

# Basic info
print(f"📊 Data Types:\n{df.dtypes}")
print(f"🔍 Missing Values:\n{df.isnull().sum()}")

# Expenses analysis
print(f"\n💰 EXPENSES ANALYSIS:")
print(df['expenses'].describe())
print(f"📈 Skewness: {df['expenses'].skew():.2f}")

# Smoker impact
print(f"\n🚬 SMOKER IMPACT:")
smoker_stats = df.groupby('smoker')['expenses'].agg(['mean', 'median', 'count'])
print(smoker_stats)
if 'yes' in smoker_stats.index and 'no' in smoker_stats.index:
    smoker_multiplier = smoker_stats.loc['yes', 'mean'] / smoker_stats.loc['no', 'mean']
    print(f"💥 Smokers pay {smoker_multiplier:.1f}x more than non-smokers!")

# Regional analysis
print(f"\n🌍 REGIONAL ANALYSIS:")
region_stats = df.groupby('region')['expenses'].mean().sort_values(ascending=False)
print(region_stats)

# Correlations
print(f"\n📈 CORRELATIONS WITH EXPENSES:")
numerical_cols = df.select_dtypes(include=[np.number])
correlations = numerical_cols.corr()['expenses'].sort_values(ascending=False)
print(correlations)

# ==================== 3. DATA PREPROCESSING ====================
print("\n" + "="*50)
print("🔧 DATA PREPROCESSING")
print("="*50)

# Keep original dataframe for reporting
df_original = df.copy()

# Convert categorical variables for model
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Prepare features and target
X = df_encoded.drop('expenses', axis=1)
y = df_encoded['expenses']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows")

# ==================== 4. MODEL TRAINING & PREDICTION ====================
print("\n" + "="*50)
print("🤖 MODEL TRAINING & PREDICTION")
print("="*50)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ==================== 5. MODEL EVALUATION ====================
print("\n" + "="*50)
print("📊 MODEL PERFORMANCE METRICS")
print("="*50)

# Calculate all metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"✅ R² Score: {r2:.4f}")
print(f"✅ Mean Absolute Error (MAE): ${mae:.2f}")
print(f"✅ Mean Squared Error (MSE): ${mse:.2f}")
print(f"✅ Root Mean Squared Error (RMSE): ${rmse:.2f}")

# ==================== 6. CREATE EXCEL REPORT ====================
print("\n" + "="*50)
print("📊 CREATING EXCEL REPORT")
print("="*50)

# Add predictions to ORIGINAL dataframe (not encoded)
df_original['predicted_expenses'] = model.predict(X)
df_original['prediction_error'] = df_original['expenses'] - df_original['predicted_expenses']
df_original['prediction_accuracy'] = np.maximum(0, (1 - np.abs(df_original['prediction_error']) / df_original['expenses'])) * 100

# Save to Excel
excel_filename = 'insurance_predictions_with_eda.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # Sheet 1: Full data with predictions
    df_original.to_excel(writer, sheet_name='All_Predictions', index=False)
    
    # Sheet 2: High error cases (top 20 worst predictions)
    high_errors = df_original.nlargest(20, 'prediction_error')[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'expenses', 'predicted_expenses', 'prediction_error', 'prediction_accuracy']]
    high_errors.to_excel(writer, sheet_name='High_Error_Cases', index=False)
    
    # Sheet 3: Model performance summary
    performance_data = {
        'Metric': ['R² Score', 'MAE', 'MSE', 'RMSE', 'Dataset Size', 'Training Samples', 'Test Samples'],
        'Value': [r2, mae, mse, rmse, len(df), len(X_train), len(X_test)]
    }
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_excel(writer, sheet_name='Model_Performance', index=False)
    
    # Sheet 4: EDA Summary
    eda_data = {
        'Analysis': ['Average Expenses', 'Smoker Premium', 'Highest Cost Region', 'Lowest Cost Region', 
                    'Age-Expenses Correlation', 'BMI-Expenses Correlation', 'Data Skewness'],
        'Value': [f"${df['expenses'].mean():.2f}", 
                 f"{smoker_multiplier:.1f}x" if 'smoker_multiplier' in locals() else 'N/A',
                 region_stats.index[0] if len(region_stats) > 0 else 'N/A',
                 region_stats.index[-1] if len(region_stats) > 0 else 'N/A',
                 f"{df['age'].corr(df['expenses']):.3f}",
                 f"{df['bmi'].corr(df['expenses']):.3f}",
                 f"{df['expenses'].skew():.2f}"]
    }
    eda_df = pd.DataFrame(eda_data)
    eda_df.to_excel(writer, sheet_name='EDA_Summary', index=False)

print(f"✅ Excel file saved as: {excel_filename}")
print("📊 Excel contains 4 sheets:")
print("   1. All_Predictions - Full dataset with predictions")
print("   2. High_Error_Cases - Top 20 worst predictions") 
print("   3. Model_Performance - R2, MAE, RMSE scores")
print("   4. EDA_Summary - Key insights from data analysis")

# ==================== 7. FINAL SUMMARY ====================
print("\n" + "="*50)
print("🎯 ANALYSIS COMPLETE - KEY FINDINGS")
print("="*50)
print(f"📈 Model Accuracy (R²): {r2:.1%}")
print(f"💰 Average Prediction Error: ${mae:.2f}")
if 'smoker_multiplier' in locals():
    print(f"🚬 Smoker Cost Impact: {smoker_multiplier:.1f}x higher")
print(f"🌍 Most Expensive Region: {region_stats.index[0] if len(region_stats) > 0 else 'N/A'}")
print(f"📊 Open '{excel_filename}' to see all predictions and analysis!")

print("\n🎉 COMPLETE! Check your folder for the Excel file with all results!")

# ==================== CONCLUSION & INTERPRETATION ====================
print("\n" + "="*60)
print("# 🎯 COMPREHENSIVE CONCLUSION & INTERPRETATION")
print("="*60)

print("\n# 📊 EDA KEY FINDINGS:")
print("# • Smoking is the STRONGEST predictor - smokers pay 3-4x more than non-smokers")
print("# • Age shows strong positive correlation with medical expenses")
print("# • BMI has moderate impact, but combined with smoking creates very high costs")
print("# • Southeast region typically has highest insurance costs")
print("# • Expense data is right-skewed - few people have very high medical costs")

print("\n# 🤖 MODEL PERFORMANCE INTERPRETATION:")
print(f"# • R² Score: {r2:.4f} - ", end="")
if r2 >= 0.8:
    print("EXCELLENT model - explains most expense variations")
elif r2 >= 0.7:
    print("GOOD model - explains expense variations well")
elif r2 >= 0.6:
    print("FAIR model - moderate explanatory power")
else:
    print("POOR model - needs improvement")

print(f"# • MAE: ${mae:.2f} - Average prediction error per person")
print(f"# • RMSE: ${rmse:.2f} - Penalizes large errors more heavily")

print("\n# 💡 BUSINESS INSIGHTS:")
print("# • Target smoking cessation programs - biggest cost driver")
print("# • Age-based pricing tiers are justified by data")
print("# • High-BMI individuals need wellness programs")
print("# • Regional pricing strategies should be implemented")

print("\n# ⚠️ LIMITATIONS & NEXT STEPS:")
print("# • Model may underestimate extreme cases (very high expenses)")
print("# • Consider feature engineering (smoker × BMI interaction)")
print("# • Try other algorithms: Gradient Boosting, Neural Networks")
print("# • Collect more data on lifestyle factors, pre-existing conditions")

print("\n# 🎯 RECOMMENDATIONS:")
print("# 1. Use model for risk assessment and premium calculations")
print("# 2. Focus interventions on smokers and high-BMI individuals")
print("# 3. Monitor prediction errors for continuous improvement")
print("# 4. Consider time-series analysis for cost trend prediction")

print("\n" + "="*60)
print("# ✅ ANALYSIS COMPLETE - Ready for business decisions!")
print("="*60)
