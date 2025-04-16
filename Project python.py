# Import necessary libraries
import pandas as pd              # For data manipulation and analysis
import numpy as np               # For numerical computations and arrays
import matplotlib.pyplot as plt  # For basic visualizations and plots
import seaborn as sns            # For advanced and beautiful visualizations

# Load the dataset
df = pd.read_csv(r"C:\Users\YADVENDRA\Desktop\archive (5)\Mental_Health_Care_in_the_Last_4_Weeks.csv")  # Load CSV data into DataFrame
df['Time Period Start Date'] = pd.to_datetime(df['Time Period Start Date'])  # Makes date-based operations easier   (time period start format ko date time format me badal diya kyuki aage ka code easy ho jaae)

# 1. Basic Info and Data Overview
print("Basic Dataset Info:")
print(df.info())                # Prints column names(non null value or dataset dikhati hai ye line)
print("\nMissing Values Count:")
print(df.isnull().sum())        # Counts missing (NaN) values for each column


# 2. Summary Statistics
print("\nSummary Statistics:")
print(df.describe())            # Shows count, mean, std, min, max, and percentiles for numeric columns


# 3. Extra Pandas EDA (Exploratory Data Analysis)


# Show descriptive stats (mean, std, min, etc.) for 'Value' grouped by 'Indicator'
indicator_dist = df.groupby('Indicator')['Value'].describe()#(isme indicator ke hisab se statics nikala hu ) 
print("\nDistribution by Indicator:")
print(indicator_dist)

# Find top 5 states with highest average value for 'Took Prescription Medication for Mental Health'
top_states = df[df['Indicator'] == 'Took Prescription Medication for Mental Health'] \
    .groupby('State')['Value'].mean().sort_values(ascending=False).head(5)
print("\nTop 5 States by Prescription Medication Usage:")
print(top_states)# jo log dawa le rahe hai uske hisab se top 5 state ka data hai

# Calculate average usage values grouped by 'Group' (e.g., By Age, By Sex, National Estimate, etc.)
group_mean = df.groupby('Group')['Value'].mean().sort_values(ascending=False)
print("\nAverage Usage by Group:")
print(group_mean)# group ke hisab se data 


# 4. Extra NumPy Analysis

# Convert value-related columns to NumPy arrays (excluding NaNs)
values = df['Value'].dropna().to_numpy()
low_ci = df['LowCI'].dropna().to_numpy()
high_ci = df['HighCI'].dropna().to_numpy()#inke nan value hata ke numpy arrays me change kar diya

# Use NumPy to calculate various stats
print("\nNumPy Statistical Summary:")
print(f"Mean Usage: {np.mean(values):.2f}%")
print(f"Median Usage: {np.median(values):.2f}%")
print(f"Standard Deviation: {np.std(values):.2f}%")
print(f"Min Usage: {np.min(values):.2f}%, Max Usage: {np.max(values):.2f}%")
print(f"Average Confidence Interval Gap: {np.mean(high_ci - low_ci):.2f}%")#ci ka difference find kiya

# Identify outliers using Z-score: values > 2 std devs away from mean
z_scores = (values - np.mean(values)) / np.std(values)
outliers = values[np.abs(z_scores) > 2]
print(f"\nOutliers Detected (Z-score > 2): {len(outliers)}")#z score wale value ko outliers mana gya



# 5. Correlation Heatmap for Numeric Columns

# Extract only numeric columns from the DataFrame
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Plot a heatmap showing correlations between numeric features
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()#sabhi numerical coloumns ke between sambandh find kiya heatmap ke dwara


# 6. National Estimate Trend Over Time

# Filter data for 'National Estimate' group
national_df = df[df['Group'] == 'National Estimate']#The trend of different indicators has been plotted over time, taking only the ‘National Estimate’ rows.

# Create a line plot for each mental health indicator over time
plt.figure(figsize=(12, 6))
for indicator in national_df['Indicator'].unique():
    subset = national_df[national_df['Indicator'] == indicator]
    plt.plot(subset['Time Period Start Date'], subset['Value'], marker='o', label=indicator)

plt.title("Mental Health Care Trends Over Time (National Estimate)")
plt.xlabel("Date")
plt.ylabel("Percentage (%)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 7. Demographic Comparison (Latest Time Period)

# Identify the latest date in the dataset
latest_date = df['Time Period Start Date'].max()#latest date par age and gender jaise row aur ciolums ko  demgraphics plot banaya

# Filter dataset for latest date and demographic groups (e.g., By Age, By Sex, etc.)
demo_df = df[(df['Time Period Start Date'] == latest_date) & (df['Group'].str.contains("By"))]

# Create a barplot comparing subgroups by mental health indicator
plt.figure(figsize=(12, 6))
sns.barplot(data=demo_df, x="Subgroup", y="Value", hue="Indicator", errorbar=None)
plt.title(f"Mental Health by Demographics ({latest_date.date()})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 8. Geographic Distribution

# Filter for state-level data for the latest date
geo_df = df[(df['Time Period Start Date'] == latest_date) & (df['Group'] == 'By State')]#state ke hisab se mental health ka graph plot

# Barplot showing mental health care access by state
plt.figure(figsize=(14, 7))
sns.barplot(data=geo_df.sort_values('Value', ascending=False), x="State", y="Value", hue="Indicator", errorbar=None)
plt.title(f"Mental Health Care Access by State ({latest_date.date()})")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# 9. Confidence Interval Analysis (Error Bars)

# Drop rows with missing confidence interval values
conf_df = df.dropna(subset=['LowCI', 'HighCI'])#confidence intervals ko error bars ke roop me plot kiya hu (bye age ke liye)

# Filter for 'By Age' group for the latest date
sample_ci = conf_df[(conf_df['Group'] == 'By Age') & (conf_df['Time Period Start Date'] == latest_date)]

# Plot error bars for each age group showing confidence intervals
plt.figure(figsize=(12, 6))
for index, row in sample_ci.iterrows():
    plt.errorbar(row['Subgroup'], row['Value'], 
                 yerr=[[row['Value'] - row['LowCI']], [row['HighCI'] - row['Value']]], 
                 fmt='o', label=row['Indicator'])

plt.title("Confidence Intervals for Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Value (%)")
plt.grid(True)
plt.tight_layout()
plt.show()



# 10. Subgroup Trend Over Time (18–29 Years)

# Filter data for only the '18 - 29 years' age group
subgroup_df = df[df['Subgroup'] == '18 - 29 years'].copy()#yeh dikhata hai kii time ke sath data kaise change hota hai

# Sort by date for time series plotting
subgroup_df = subgroup_df.sort_values('Time Period Start Date')

# Plot line chart for each indicator for this age group over time
plt.figure(figsize=(12, 6))
for indicator in subgroup_df['Indicator'].unique():
    sub = subgroup_df[subgroup_df['Indicator'] == indicator]
    plt.plot(sub['Time Period Start Date'], sub['Value'], marker='o', label=indicator)

plt.title("Trend for 18–29 Year Olds Over Time")
plt.xlabel("Date")
plt.ylabel("Value (%)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 11. Final Correlation Heatmap (Clean Numeric Data Only)

# Select only numeric columns again
numeric_df = df.select_dtypes(include=np.number)

# Drop completely empty columns and rows with missing values
numeric_df = numeric_df.dropna(axis=1, how='all')
clean_numeric_df = numeric_df.dropna()#data ka nan hatakar correlation heatmap banaya5

# Plot heatmap if data exists after cleanup
if not clean_numeric_df.empty:
    plt.figure(figsize=(10, 6))
    sns.heatmap(clean_numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.show()
else:
    print("No usable numeric data available for correlation heatmap.")
