
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas.plotting import andrews_curves


df = pd.read_csv('Titanic_Train.csv')

if not df.empty:
    print("Dataset loaded successfully.")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()

    df['Age'].fillna(df['Age'].median(), inplace=True)

   
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    
    df.drop('Cabin', axis=1, inplace=True)


    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)


    print("\n--- Question 1: Survival Percentage ---")
    survival_percentage = df['Survived'].value_counts(normalize=True) * 100
    print(f"Survival Percentage:\n{survival_percentage}\n")

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='Survived', data=df, palette='viridis')
    plt.title('Survival Count of Titanic Passengers', fontsize=16)
    plt.xlabel('Survival Status (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Number of Passengers', fontsize=12)
    plt.xticks([0, 1], ['Did not Survive', 'Survived'])
  
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
    plt.show()


    print("\n--- Question 2: Gender and Survival ---")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Sex', hue='Survived', data=df, palette='plasma')
    plt.title('Survival Count by Gender', fontsize=16)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Number of Passengers', fontsize=12)
    plt.legend(title='Survival Status', labels=['Did not Survive', 'Survived'])
    plt.show()


    print("\n--- Question 3: Passenger Class and Survival ---")
    class_survival = df.groupby(['Pclass', 'Survived']).size().unstack()
    class_survival.plot(kind='bar', stacked=True, color=['#440154', '#21908d'])
    plt.title('Survival Count by Passenger Class (Stacked)', fontsize=16)
    plt.xlabel('Passenger Class', fontsize=12)
    plt.ylabel('Number of Passengers', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Survival Status', labels=['Did not Survive', 'Survived'])
    plt.show()


    print("\n--- Question 4: Age and Survival ---")
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', kde=True, bins=30, palette='magma')
    plt.title('Age Distribution by Survival Status', fontsize=16)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Number of Passengers', fontsize=12)
    plt.legend(title='Survival Status', labels=['Survived', 'Did not Survive'])
    plt.show()

    print("\n--- Question 5: Age Group Survival Rate ---")
    age_bins = [0, 12, 18, 35, 60, 80]
    age_labels = ['Child', 'Teenager', 'Youth', 'Adult', 'Elderly']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

    age_group_survival = df.groupby('AgeGroup')['Survived'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='AgeGroup', y='Survived', data=age_group_survival, palette='coolwarm')
    plt.title('Survival Rate by Age Group', fontsize=16)
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.ylim(0, 1)
    plt.show()



    print("\n--- Question 6: Family Size and Survival ---")
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    family_survival = df.groupby('FamilySize')['Survived'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='FamilySize', y='Survived', data=family_survival, palette='viridis')
    plt.title('Survival Rate by Family Size', fontsize=16)
    plt.xlabel('Family Size (SibSp + Parch + 1)', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.ylim(0, 1)
    plt.show()
   

    print("\n--- Question 7: Embarkation Port and Survival ---")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Embarked', hue='Survived', data=df, palette='Set2')
    plt.title('Survival Count by Port of Embarkation', fontsize=16)
    plt.xlabel('Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)', fontsize=12)
    plt.ylabel('Number of Passengers', fontsize=12)
    plt.legend(title='Survival Status', labels=['Did not Survive', 'Survived'])
    plt.show()



    print("\n--- Question 8: Fare and Survival (Quartile/Box Plot) ---")
    plt.figure(figsize=(12, 7))
    # Using a log scale for the y-axis to better visualize the distribution for a wide range of fares
    sns.boxplot(x='Survived', y='Fare', data=df, palette='winter')
    plt.title('Fare Distribution by Survival Status', fontsize=16)
    plt.xlabel('Survival Status (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Fare', fontsize=12)
    plt.xticks([0, 1], ['Did not Survive', 'Survived'])
    # plt.yscale('log') # Uncomment for log scale if fares are heavily skewed
    plt.show()
    


    print("\n--- Question 9: Combination of Factors (Gender + Class) ---")
    g = sns.FacetGrid(df, col='Pclass', row='Sex', hue='Survived', margin_titles=True, palette='seismic')
    g.map(plt.hist, 'Age', bins=20, alpha=0.7)
    g.add_legend(title='Survival Status', labels=['Did not Survive', 'Survived'])
    g.fig.suptitle('Survival by Age, Gender, and Class', y=1.03, fontsize=16)
    plt.show()
    


   
    print("\n--- Question 10: Missing Data Analysis ---")

    original_df = pd.read_csv('Titanic_Train.csv')
    missing_data = original_df.isnull().sum().sort_values(ascending=False)
    print("Count of missing values per column:")
    print(missing_data[missing_data > 0])

    plt.figure(figsize=(10, 6))
    sns.heatmap(original_df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Heatmap of Missing Data in the Titanic Dataset', fontsize=16)
    plt.show()


    print("\n--- Additional Requested Visualizations ---")

   
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', style='Pclass', palette='bwr', alpha=0.8)
    plt.title('Scatterplot of Age vs. Fare by Survival and Class', fontsize=16)
    plt.show()
    print("Findings (Scatter Multiple): Shows that most non-survivors are clustered in the low-fare, young-adult area. Survivors are more spread out, especially towards higher fares.")

   
    sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare', 'FamilySize']], hue='Survived', palette='viridis')
    plt.suptitle('Scatter Matrix (Pairplot) of Key Features', y=1.02, fontsize=16)
    plt.show()
    print("Findings (Scatter Matrix): Provides a matrix of relationships. The histograms on the diagonal re-confirm previous findings (e.g., more non-survivors). The scatterplots show correlations, such as Fare being inversely related to Pclass.")

    fig = px.scatter(df, x="Age", y="Fare",
                 size="FamilySize", color="Survived",
                 hover_name="Name", log_y=True, size_max=60,
                 title="Bubble Chart: Age vs Fare (Size by Family Size, Color by Survival)")
    fig.show()
    print("Findings (Bubble Chart): An interactive plot where bubble size represents family size. It highlights that many of the largest families (largest bubbles) with low fares did not survive.")

  
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=df, x='Age', y='Fare', hue='Survived', fill=True, palette='coolwarm')
    plt.title('2D Density Chart of Age and Fare by Survival', fontsize=16)
    plt.ylim(0, 300)  
    plt.show()
    print("Findings (Density Chart): The red area (not survived) shows a high concentration of passengers around 20-30 years old who paid very low fares. The blue area (survived) is more spread out, with a notable concentration at higher fare levels.")

    parallel_df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Survived']]
    parallel_df['Sex'] = parallel_df['Sex'].astype('category').cat.codes
    parallel_df['Embarked'] = parallel_df['Embarked'].astype('category').cat.codes

    fig = px.parallel_coordinates(parallel_df, color="Survived",
                                  color_continuous_scale=px.colors.sequential.Viridis,
                                  title="Parallel Coordinates Plot for Survival Factors")
    fig.show()
    print("Findings (Parallel Coordinates): This chart maps each passenger as a line. A large bundle of blue lines (survived) can be seen passing through Sex=0 (female) and Pclass=1. A large bundle of yellow lines (not survived) passes through Sex=1 (male) and Pclass=3.")


    avg_survival_rate = df['Survived'].mean()
    class_survival_rate = df.groupby('Pclass')['Survived'].mean()
    class_survival_rate['Average'] = avg_survival_rate
    deviation = (class_survival_rate - avg_survival_rate).sort_values(ascending=False)

    deviation.plot(kind='barh', color=(deviation > 0).map({True: 'g', False: 'r'}))
    plt.title('Deviation of Survival Rate by Class from Average', fontsize=16)
    plt.xlabel('Deviation from Average Survival Rate')
    plt.axvline(0, color='k', linestyle='--')
    plt.show()
    print("Findings (Deviation Chart): Clearly shows that 1st class passengers had a survival rate well above the average, while 3rd class passengers were significantly below average.")

 
    plt.figure(figsize=(12, 8))
    andrews_curves(parallel_df, 'Survived', colormap='viridis')
    plt.title('Andrews Curves of Titanic Features by Survival', fontsize=16)
    plt.show()
    print("Findings (Andrews Curves): Andrews curves represent each data point as a curve. Different classes (colors) show different patterns. The green curves (survived) tend to have a different shape and grouping from the purple curves (not survived), indicating that the underlying feature values can be used to separate the two outcomes.")