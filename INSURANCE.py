import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv('/content/insurance (1).csv')

expenses_skew = round(df['expenses'].skew(), 2)

smoker_ratio = round(
    df.groupby('smoker')['expenses'].mean()['yes'] /
    df.groupby('smoker')['expenses'].mean()['no'], 1
)

age_corr = round(df['age'].corr(df['expenses']), 3)
bmi_corr = round(df['bmi'].corr(df['expenses']), 3)

df_encoded = pd.get_dummies(
    df,
    columns=['sex', 'smoker', 'region'],
    drop_first=True
)

X = df_encoded.drop('expenses', axis=1)
y = df_encoded['expenses']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

r2 = round(r2_score(y_test, y_test_pred), 3)
mae = round(mean_absolute_error(y_test, y_test_pred), 2)
rmse = round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2)

df['Predicted_Expenses'] = model.predict(X)
df['Prediction_Error'] = df['expenses'] - df['Predicted_Expenses']

summary = pd.DataFrame({
    'Metric': [
        'Expenses Skewness',
        'Smoker Expense Ratio',
        'R2 Score',
        'Mean Absolute Error',
        'Root Mean Squared Error',
        'Age vs Expenses Correlation',
        'BMI vs Expenses Correlation'
    ],
    'Value': [
        expenses_skew,
        smoker_ratio,
        r2,
        mae,
        rmse,
        age_corr,
        bmi_corr
    ]
})

with pd.ExcelWriter('Insurance_Analytics_Project.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Predictions', index=False)
    df.loc[
        df['Prediction_Error'].abs().nlargest(20).index
    ].to_excel(writer, sheet_name='High_Error_Cases', index=False)
    summary.to_excel(writer, sheet_name='Model_Summary', index=False)

plt.figure(figsize=(6,4))
plt.hist(df['expenses'], bins=50)
plt.title('Distribution of Medical Expenses')
plt.tight_layout()
plt.savefig('expenses_distribution.png')
plt.close()

plt.figure(figsize=(6,4))
df.boxplot(column='expenses', by='smoker')
plt.title('Expenses by Smoking Status')
plt.suptitle('')
plt.tight_layout()
plt.savefig('smoker_vs_expenses.png')
plt.close()

plt.figure(figsize=(6,5))
plt.scatter(df['expenses'], df['Predicted_Expenses'], alpha=0.5)
plt.plot(
    [df['expenses'].min(), df['expenses'].max()],
    [df['expenses'].min(), df['expenses'].max()]
)
plt.xlabel('Actual Expenses')
plt.ylabel('Predicted Expenses')
plt.title('Actual vs Predicted Expenses')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()
