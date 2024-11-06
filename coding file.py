import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Load the data
data = pd.read_csv('DataScience_salaries_2024.csv')

# Display basic information about the dataset
data_info = data.info()
data_head = data.head()

# Step 1: Statistical Analysis
# Descriptive statistics
data_description = data.describe()

# Selecting only numeric columns for correlation, skewness, and kurtosis calculations
numeric_data = data.select_dtypes(include=['number'])

# Correlation matrix
data_correlation = numeric_data.corr()

# Skewness and Kurtosis
data_skewness = numeric_data.apply(skew)
data_kurtosis = numeric_data.apply(kurtosis)

# Step 2: Visualization functions

# Bar Chart - average salary by job title
def plot_bar():
    if 'job_title' in data.columns and 'salary' in data.columns:
        # Calculate the average salary for each job title
        avg_salary_by_job = data.groupby('job_title')['salary'].mean().sort_values(ascending=False).head(10)
        
        # Plot the bar chart
        plt.figure(figsize=(12, 6))
        sns.barplot(x=avg_salary_by_job.index, y=avg_salary_by_job.values)
        plt.title('Average Salary by Job Title (Top 10)')
        plt.xlabel('Job Title')
        plt.ylabel('Average Salary')
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("Columns 'job_title' or 'salary' not found for bar chart.")

# Pie Chart - showing top N job titles only
def plot_pie_top_n(n=5):
    if 'job_title' in data.columns:
        # Count the occurrences of each job title and separate the top N
        top_job_titles = data['job_title'].value_counts()
        top_n = top_job_titles.nlargest(n)
        others_count = top_job_titles[n:].sum()
        
        # Create a new series with the top N and the "Other" category
        top_n['Other'] = others_count
        
        # Plot the pie chart
        plt.figure(figsize=(8, 8))
        top_n.plot.pie(autopct='%1.1f%%')
        plt.title(f'Top {n} Job Title Distribution with "Other" Category')
        plt.ylabel('')
        plt.show()
    else:
        print("Column 'job_title' not found for pie chart.")

# Heatmap - correlation heatmap
def plot_heatmap():
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

# Box Plot - salary distribution by job title or other relevant grouping
def plot_box():
    if 'salary' in data.columns and 'job_title' in data.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='job_title', y='salary', data=data)
        plt.xticks(rotation=45)
        plt.title('Salary Distribution by Job Title')
        plt.xlabel('Job Title')
        plt.ylabel('Salary')
        plt.show()
    else:
        print("Columns 'salary' or 'job_title' not found for box plot.")

# Scatter Plot - relationship between experience level and salary
def plot_scatter():
    if 'experience_level' in data.columns and 'salary' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='experience_level', y='salary', data=data)
        plt.title('Salary vs Experience Level')
        plt.xlabel('Experience Level')
        plt.ylabel('Salary')
        plt.show()
    else:
        print("Columns 'experience_level' or 'salary' not found for scatter plot.")

# Display statistical analysis
print("Data Info:\n", data_info)
print("Data Head:\n", data_head)
print("Descriptive Statistics:\n", data_description)
print("Correlation Matrix:\n", data_correlation)
print("Skewness:\n", data_skewness)
print("Kurtosis:\n", data_kurtosis)

# Call each plotting function
plot_bar()            # Bar chart showing average salary by job title
plot_pie_top_n(n=5)   # Pie chart with top 5 job titles and "Other" category
plot_heatmap()        # Correlation heatmap for numerical columns
plot_scatter()        # Scatter plot for salary vs experience level