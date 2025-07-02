import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tabulate import tabulate


data = pd.read_csv('f1_pitstops_2018_2024.csv')
data = data.dropna()

data['Position'] = data['Position'].astype(int)
data['Tire Compound'] = data['Tire Compound'].map({
    'ULTRASOFT': 1, 'SOFT': 2, 'SUPERSOFT': 3, 'MEDIUM': 4, 'HARD': 5
})
data = data[data['Laps'] > 0]

y = (data['Position'] == 1).astype(int)
count_0 = y.value_counts()[0]
count_1 = y.value_counts()[1]
print(f"\n Class Distribution:\n Winner [1]: {count_1}\n Others [0]: {count_0}")

features = ['AvgPitStopTime', 'Tire Usage Aggression', 'Driver Aggression Score', 'Lap Time Variation']
X = data[features]

sns.pairplot(pd.concat([X, y.rename("Target")], axis=1), hue="Target", palette='muted')
plt.suptitle("Pairwise Feature Visualization", y=1.02, fontsize=14)
plt.show()

plt.figure(figsize=(14, 10))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, fmt=".2f", cmap='Blues', linewidths=0.5, linecolor='white', annot_kws={'fontsize': 8}, cbar_kws={'shrink': 0.9, 'aspect': 20})
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'Accuracy': accuracy, 'F1 Score': f1}

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 10}, cbar_kws={'shrink': 0.9, 'aspect': 20})
    plt.title(f'Confusion Matrix - {name}', fontsize=12)
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('Actual', fontsize=10)
    plt.tight_layout()
    plt.show()

    true_neg = ((y_test == 0) & (y_pred == 0)).sum()
    false_pos = ((y_test == 0) & (y_pred == 1)).sum()
    false_neg = ((y_test == 1) & (y_pred == 0)).sum()
    true_pos = ((y_test == 1) & (y_pred == 1)).sum()

results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ['Model', 'Accuracy', 'F1 Score']
print("\n Final Model Comparison Table:")
print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False))

best_model = max(results, key=lambda x: results[x]['Accuracy'])
print(f"\n Best Model: {best_model} with Accuracy: {results[best_model]['Accuracy']:.2f}")

final_model = models[best_model]
champion_probs = final_model.predict_proba(X_test)[:, 1]
champion_index = np.argmax(champion_probs)
champion_driver = data.loc[X_test_df.index[champion_index], 'Driver']
print(f"\n Predicted Formula 1 Champion (2025): {champion_driver}")