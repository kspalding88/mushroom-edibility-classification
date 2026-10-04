# =============================================================================
# Fungal Forecast - Predicting Mushroom Edibility from Observable Traits
# Author: Kyle M. Spalding
# DSC680: Applied Data Science
# =============================================================================
# Goal: predict whether a mushroom is edible or poisonous from its features.
# We compare a few models and check how well we do using only the traits a
# beginner forager could actually see in the field.
# =============================================================================

# Import needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

import os
import joblib

sns.set_style("whitegrid")

# Resolve paths relative to this script so the project runs anywhere it is cloned.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder to save figures for the white paper
fig_dir = os.path.join(BASE_DIR, "figures")
os.makedirs(fig_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load the data
# -----------------------------------------------------------------------------
# The file is semicolon-separated, so we tell pandas with sep=';'
data_path = os.path.join(BASE_DIR, "data", "secondary_data.csv")
mushrooms = pd.read_csv(data_path, sep=';')

print(f"Total records loaded: {len(mushrooms):,}")
print(f"Columns: {len(mushrooms.columns)}")
print(mushrooms.head())

# -----------------------------------------------------------------------------
# 2. Explore the data
# -----------------------------------------------------------------------------
# Check how much data is missing in each column
print("\n*Missing Data*")
print(mushrooms.isnull().sum().sort_values(ascending=False))

# Show missing data as a PERCENTAGE of all rows. These are the numbers quoted in
# the paper (e.g., veil-type ~95%, spore-print-color ~90%) and the reason we
# decide to drop the columns that are more than 80% empty.
missing_pct = (mushrooms.isnull().mean() * 100).sort_values(ascending=False)
print("\n*Missing Data (% of rows)*")
print(missing_pct[missing_pct > 0].round(1))

# Count exact duplicate rows. The paper reports 146 duplicates; this line proves it.
print("\nExact duplicate rows:", mushrooms.duplicated().sum())

# Check the balance of edible vs poisonous ('p' = poisonous, 'e' = edible)
print("\n=== Class Balance ===")
print(mushrooms['class'].value_counts())
# Same balance as percentages (paper quotes ~55.5% poisonous, ~44.5% edible)
print(mushrooms['class'].value_counts(normalize=True).mul(100).round(1))

# -----------------------------------------------------------------------------
# 3. Clean the data
# -----------------------------------------------------------------------------
# Some columns are almost entirely empty. Filling them in would be mostly guessing,
# so we just drop the columns that are more than 80% missing.
too_empty = ['stem-root', 'veil-type', 'veil-color', 'spore-print-color']
mushrooms = mushrooms.drop(columns=too_empty)

# For the columns that still have a few blanks, fill them with the word "unknown"
# so we keep every row instead of throwing data away.
mushrooms = mushrooms.fillna('unknown')

print("\nMissing values left after cleaning:", mushrooms.isnull().sum().sum())

# -----------------------------------------------------------------------------
# 4. A couple of quick charts
# -----------------------------------------------------------------------------
# Chart 1: how many edible vs poisonous mushrooms
labels = mushrooms['class'].map({'p': 'Poisonous', 'e': 'Edible'})
counts = labels.value_counts()

plt.figure(figsize=(7, 5))
bars = plt.bar(counts.index, counts.values, color=['#66c2a5', '#fc8d62'])
plt.title('Number of Edible vs Poisonous Mushrooms')
plt.xlabel('Mushroom Type')
plt.ylabel('Number of Mushrooms')
# Write the count on top of each bar so the chart is easy to read
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 300,
             f'{height:,}', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig1_class_balance.png'), dpi=150)
plt.close()

# Chart 2: cap diameter for each class (do poisonous ones tend to be bigger?)
plt.figure(figsize=(7, 5))
sns.boxplot(data=mushrooms.assign(edibility=labels),
            x='edibility', y='cap-diameter', showfliers=False)
plt.title('Cap Diameter by Edibility')
plt.xlabel('Mushroom Type')
plt.ylabel('Cap Diameter (cm)')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig2_cap_diameter.png'), dpi=150)
plt.close()

# -----------------------------------------------------------------------------
# 5. Get the data ready for modeling
# -----------------------------------------------------------------------------
# y is what we want to predict: 1 = poisonous, 0 = edible.
# We make poisonous = 1 because catching poisonous mushrooms is the important part.
y = (mushrooms['class'] == 'p').astype(int)

# X is everything except the answer column.
# get_dummies turns text categories (like cap-color) into 0/1 columns the models can use.
X = pd.get_dummies(mushrooms.drop(columns=['class']))

# Split into training data (to learn from) and test data (to check ourselves on)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"\nTrain rows: {len(X_train):,}   Test rows: {len(X_test):,}")

# -----------------------------------------------------------------------------
# 6. Train and compare three models
# -----------------------------------------------------------------------------
# We try three models from simple to more powerful:
#   Logistic Regression - simple starting point
#   Decision Tree       - makes easy-to-read yes/no rules
#   Random Forest       - many trees combined, usually the most accurate

# Model 1: Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
logreg_predictions = logreg.predict(X_test)

print("\n===== Logistic Regression =====")
print(f"Accuracy: {accuracy_score(y_test, logreg_predictions):.3f}")
print(f"Poisonous caught (recall): {recall_score(y_test, logreg_predictions):.3f}")

# Model 2: Decision Tree
tree = DecisionTreeClassifier(max_depth=8, random_state=42)
tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test)

print("\n===== Decision Tree =====")
print(f"Accuracy: {accuracy_score(y_test, tree_predictions):.3f}")
print(f"Poisonous caught (recall): {recall_score(y_test, tree_predictions):.3f}")

# Model 3: Random Forest
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)
forest_predictions = forest.predict(X_test)

print("\n===== Random Forest =====")
print(f"Accuracy: {accuracy_score(y_test, forest_predictions):.3f}")
print(f"Poisonous caught (recall): {recall_score(y_test, forest_predictions):.3f}")

# -----------------------------------------------------------------------------
# 7. Look closer at the best model (Random Forest)
# -----------------------------------------------------------------------------
# A confusion matrix shows exactly which guesses were right and wrong
cm = confusion_matrix(y_test, forest_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=',d', cmap='YlOrRd',
            xticklabels=['Edible', 'Poisonous'],
            yticklabels=['Edible', 'Poisonous'])
plt.title('Random Forest - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig3_confusion_matrix.png'), dpi=150)
plt.close()

# The dangerous mistake: a poisonous mushroom we called edible
print("\nDangerous misses (poisonous called edible):", cm[1, 0])

# -----------------------------------------------------------------------------
# 7b. Safety-first threshold tuning (Research Question 4)
# -----------------------------------------------------------------------------
# By default a model calls a mushroom "poisonous" only when it is more than 50%
# sure. In a life-safety setting that is too relaxed: we would rather get a few
# false alarms than miss a single poisonous mushroom. So we lower the cutoff and
# watch the tradeoff between "poisonous caught" and "edibles wrongly flagged".

# Probability the Random Forest gives to the "poisonous" class for each mushroom
poison_proba = forest.predict_proba(X_test)[:, 1]

# AUC summarizes how well the model ranks poisonous above edible (1.0 = perfect)
auc = roc_auc_score(y_test, poison_proba)
print(f"\nRandom Forest ROC-AUC: {auc:.3f}")

# Try several cutoffs from strict (0.50) to very cautious (0.05).
# For each one we count how many poisonous we caught and how many edibles we
# wrongly flagged as poisonous.
print("\n===== Threshold Tradeoff (Safety Tuning) =====")
edible_count = (y_test == 0).sum()
cutoffs = [0.50, 0.30, 0.20, 0.10, 0.05]
for cutoff in cutoffs:
    preds = (poison_proba >= cutoff).astype(int)
    caught = recall_score(y_test, preds)
    misses = ((preds == 0) & (y_test == 1)).sum()      # poisonous called edible
    false_alarms = ((preds == 1) & (y_test == 0)).sum()  # edible called poisonous
    print(f"Cutoff {cutoff:.2f}: poison caught {caught:.3f}, "
          f"dangerous misses {misses}, edibles flagged {false_alarms}")

# ROC curve figure: shows the tradeoff between catching poison (true positive
# rate) and false alarms (false positive rate) across every possible cutoff.
fpr, tpr, thresholds = roc_curve(y_test, poison_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='#c44e52', label=f'Random Forest (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random guessing')
plt.xlabel('False Alarm Rate (edibles flagged as poisonous)')
plt.ylabel('Poisonous Caught (true positive rate)')
plt.title('ROC Curve - Safety Tradeoff')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig5_roc_curve.png'), dpi=150)
plt.close()

# -----------------------------------------------------------------------------
# 8. Novice test: use only features a beginner can easily see
# -----------------------------------------------------------------------------
# Some traits are hard for a beginner to judge. Here we keep only the easy,
# obvious ones and see if the model is still accurate.
easy_features = ['cap-diameter', 'stem-height', 'stem-width', 'cap-shape',
                 'cap-color', 'does-bruise-or-bleed', 'stem-color',
                 'has-ring', 'habitat', 'season']

X_easy = pd.get_dummies(mushrooms[easy_features])
Xe_train, Xe_test, ye_train, ye_test = train_test_split(
    X_easy, y, test_size=0.25, random_state=42
)

easy_model = RandomForestClassifier(n_estimators=200, random_state=42)
easy_model.fit(Xe_train, ye_train)
easy_predictions = easy_model.predict(Xe_test)

print("\n===== Beginner-Features-Only Model =====")
print(f"Accuracy: {accuracy_score(ye_test, easy_predictions):.3f}")
print(f"Poisonous caught (recall): {recall_score(ye_test, easy_predictions):.3f}")

# -----------------------------------------------------------------------------
# 9. Which features matter most?
# -----------------------------------------------------------------------------
# The Random Forest can tell us which features it relied on the most.
importances = pd.Series(forest.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(9, 6))
top10.sort_values().plot(kind='barh', color='#4c72b0')
plt.title('Top 10 Most Important Features')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig4_feature_importance.png'), dpi=150)
plt.close()

print("\n=== Top 10 Features ===")
print(top10)

# -----------------------------------------------------------------------------
# 10. Save the model so it can be reused later (no retraining needed)
# -----------------------------------------------------------------------------
# We save the beginner-features model because that's the one a real tool would
# use (it only needs traits a normal person can see). We also save the list of
# columns, because a new mushroom has to be encoded the exact same way.
joblib.dump(easy_model, os.path.join(fig_dir, 'mushroom_model.pkl'))
joblib.dump(list(X_easy.columns), os.path.join(fig_dir, 'model_columns.pkl'))
print("\nModel saved to mushroom_model.pkl")

# -----------------------------------------------------------------------------
# 11. Example: how a tool would use the saved model on ONE new mushroom
# -----------------------------------------------------------------------------
# TO CHECK A MUSHROOM YOU FOUND: edit the values below to match what you see,
# then run the script. The model uses single-letter codes, so use the legend
# in the comment next to each line to pick the right letter.
#
# The three numbers are measured with a ruler:
#   cap-diameter -> width of the cap, in centimeters (e.g., 8.0)
#   stem-height  -> height of the stem, in centimeters (e.g., 6.5)
#   stem-width   -> thickness of the stem, in millimeters (e.g., 15.0)
#
# Everything else is a code you look up below. Just replace the letter.

new_mushroom = {
    # --- Measurements (numbers) ---
    'cap-diameter': 8.0,   # cap width in cm
    'stem-height': 6.5,    # stem height in cm
    'stem-width': 15.0,    # stem thickness in mm

    # --- Cap shape: b=bell  c=conical  f=flat  o=others  p=sunken  s=spherical  x=convex
    'cap-shape': 'x',

    # --- Cap color: b=buff  e=red  g=gray  k=black  l=blue  n=brown  o=orange
    #                p=pink  r=green  u=purple  w=white  y=yellow
    'cap-color': 'n',

    # --- Does it bruise or bleed when handled?  f=no  t=yes
    'does-bruise-or-bleed': 'f',

    # --- Stem color: b=buff  e=red  f=none  g=gray  k=black  l=blue  n=brown
    #                 o=orange  p=pink  r=green  u=purple  w=white  y=yellow
    'stem-color': 'w',

    # --- Does it have a ring on the stem?  f=no  t=yes
    'has-ring': 't',

    # --- Habitat (where it grows): d=woods  g=grasses  h=heaths  l=leaves
    #                m=meadows  p=paths  u=urban  w=waste
    'habitat': 'd',

    # --- Season: a=autumn  s=spring  u=summer  w=winter
    'season': 'a'
}

# Load the saved model and column list back (this is what an app would do)
saved_model = joblib.load(os.path.join(fig_dir, 'mushroom_model.pkl'))
saved_columns = joblib.load(os.path.join(fig_dir, 'model_columns.pkl'))

# Turn the one mushroom into a single-row table and encode it the same way.
new_df = pd.DataFrame([new_mushroom])
new_encoded = pd.get_dummies(new_df)

# The new mushroom probably won't have every column the model was trained on,
# so we add any missing columns and set them to 0, then put them in the same order.
for col in saved_columns:
    if col not in new_encoded.columns:
        new_encoded[col] = 0
new_encoded = new_encoded[saved_columns]

# Get the chance the mushroom is poisonous (a number between 0 and 1).
poison_chance = saved_model.predict_proba(new_encoded)[0][1]

# Instead of only saying "edible" or "poisonous", we add a third "NOT CERTAIN"
# group for the cases in the middle where the model is not confident. This is
# safer, because a borderline mushroom should be treated as risky, not eaten.
#   - poison chance above 60%  -> POISONOUS
#   - poison chance below 40%  -> EDIBLE
#   - anything in between        -> NOT CERTAIN (treat as high risk)
if poison_chance >= 0.60:
    label = 'POISONOUS'
elif poison_chance <= 0.40:
    label = 'EDIBLE'
else:
    label = 'NOT CERTAIN'

print("\n=== Example Prediction for One Mushroom ===")
print(f"Chance poisonous: {poison_chance:.0%}")
print(f"Prediction: {label}")
if label == 'NOT CERTAIN':
    print("The model is not confident about this one. Treat it as HIGH RISK.")
print("SAFETY NOTE: This is a decision-support tool only. Never eat a wild")
print("mushroom based on this result. Always confirm with an expert.")

print("\nAll figures saved to:", fig_dir)
print("Done.")
