import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


df = pd.read_csv("Cheek - pinch skin.csv")


demographics = ['adult_child', 'age', 'sex', 'handedness',
                'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']

# selecting numerical stuff only
reading_cols = [
    col for col in df.columns
    if col not in ['sequence_id', 'sequence_counter', 'behavior', 'phase'] + demographics
    and pd.api.types.is_numeric_dtype(df[col])
]

# summarize each sequence 
seq_summary = []
for seq_id, group in df.groupby('sequence_id'):
    group = group.sort_values('sequence_counter')  # keeps correct order VERY IMPORTANT
    summary = {col: group[col].mean() for col in reading_cols}  # average readings
    # add demographics (same in sequence)
    for demo in demographics:
        summary[demo] = group[demo].iloc[0]
    summary['sequence_id'] = seq_id
    seq_summary.append(summary)

seq_df = pd.DataFrame(seq_summary)

#correlation heatmap 
corr_matrix = seq_df[reading_cols + demographics].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix.loc[demographics, reading_cols], cmap='coolwarm', annot=False)
plt.title("Correlation between Demographics and Readings")
plt.tight_layout()
plt.show()


# age vs a sample reading (acc_x)
if 'acc_x' in seq_df.columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=seq_df, x='age', y='acc_x', hue='sex', palette='Set1')
    sns.regplot(data=seq_df, x='age', y='acc_x', scatter=False, color='blue')
    plt.title("Age vs acc_x")
    plt.tight_layout()
    plt.show()

# calculate raw acceleration magnitude for every row
df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

# scatter plot: age vs raw Acceleration magnitude (all rows)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='acc_magnitude', hue='sex', palette='coolwarm', s=10, alpha=0.6)
plt.title("Age vs Raw Acceleration Magnitude (all readings)")
plt.xlabel("Age")
plt.ylabel("Acceleration Magnitude")
plt.legend(title="Sex", labels=["Male", "Female"])
plt.tight_layout()
plt.show()


# statistical tests
if 'acc_x' in seq_df.columns:
    male = seq_df[seq_df['sex'] == 0]['acc_x']
    female = seq_df[seq_df['sex'] == 1]['acc_x']
    t_stat, p_val = stats.ttest_ind(male, female)
    print(f"T-test for sex effect on acc_x: t={t_stat:.3f}, p={p_val:.3f}")

#save summary for later inspection
seq_df.to_csv("output/sequence_summary.csv", index=False)

seq_df.head()
