import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reading the training data
train_data = pd.read_csv('./IE425_Spring25_train_data.csv')

# Constructing male/female brand lists
female_brands = []
grouped = train_data.groupby("brand_name")
total_user_counts = grouped["unique_id"].nunique()
female_user_counts = train_data[train_data["gender"] == "F"].groupby("brand_name")["unique_id"].nunique()
percentage_female = (female_user_counts / total_user_counts) * 100
high_female_brands = percentage_female[(percentage_female >= 90) & (female_user_counts >= 10)]
female_brands = high_female_brands.index.tolist()

male_brands = []
grouped = train_data.groupby("brand_name")
total_user_counts = grouped["unique_id"].nunique()
male_user_counts = train_data[train_data["gender"] == "M"].groupby("brand_name")["unique_id"].nunique()
percentage_male = (male_user_counts / total_user_counts) * 100
high_male_brands = percentage_male[(percentage_male >= 80) & (male_user_counts >= 5)]
male_brands = high_male_brands.index.tolist()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


# Preprocessing
df_copy = train_data.copy()
df_copy['product_name'] = df_copy['product_name'].fillna("").str.lower()
df_copy['gender'] = df_copy['gender'].fillna("").str.lower()

# Filter for female products
male_data = df_copy[df_copy['gender'] == 'm']

# Step 1: Vectorize product names
vectorizer = CountVectorizer(min_df=3, stop_words='english')  # min_df=3 removes rare words
male_matrix = vectorizer.fit_transform(male_data['product_name'])
male_words = vectorizer.get_feature_names_out()

# Step 2: Reduce dimensionality
svd = TruncatedSVD(n_components=50, random_state=42)
male_reduced = svd.fit_transform(male_matrix.T)  # Transpose to cluster words

# Step 3: KMeans clustering
n_clusters = 70
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
male_labels = kmeans.fit_predict(male_reduced)

# Step 4: Group words by cluster
male_cluster_df = pd.DataFrame({
    'word': male_words,
    'cluster': male_labels
})

# Display example words per cluster
for i in range(n_clusters):
    print(f"\n🔸 Cluster {i} (male):")
    print(male_cluster_df[male_cluster_df['cluster'] == i]['word'].head(10).tolist())


female_data = df_copy[df_copy['gender'] == 'f']

# Step 1: Vectorize product names
vectorizer = CountVectorizer(min_df=3, stop_words='english')  # min_df=3 removes rare words
female_matrix = vectorizer.fit_transform(female_data['product_name'])
female_words = vectorizer.get_feature_names_out()

# Step 2: Reduce dimensionality
svd = TruncatedSVD(n_components=50, random_state=42)
female_reduced = svd.fit_transform(female_matrix.T)  # Transpose to cluster words

# Step 3: KMeans clustering
n_clusters = 70
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
female_labels = kmeans.fit_predict(female_reduced)

# Step 4: Group words by cluster
female_cluster_df = pd.DataFrame({
    'word': female_words,
    'cluster': female_labels
})

# Display example words per cluster
for i in range(n_clusters):
    print(f"\n🔸 Cluster {i} (female):")
    print(female_cluster_df[female_cluster_df['cluster'] == i]['word'].head(10).tolist())


# Truncate male clusters to first 10 words
male_keyword_dict = {
    f"cluster_{i}": male_cluster_df[male_cluster_df['cluster'] == i]['word'].tolist()[:10]
    for i in range(0, 65)
}

# Truncate female clusters to first 10 words
female_keyword_dict = {
    f"cluster_{i}": female_cluster_df[female_cluster_df['cluster'] == i]['word'].tolist()[:10]
    for i in range(0, 65)
}

# Print sample output
print("✅ Male Clusters (Top 10):")
for k, v in male_keyword_dict.items():
    print(f"{k}: {v}")

print("\n✅ Female Clusters (Top 10):")
for k, v in female_keyword_dict.items():
    print(f"{k}: {v}")


# Rebuild reverse mapping for each word → cluster number
male_word_to_cluster = {
    word: int(cluster.replace("cluster_", ""))
    for cluster, words in male_keyword_dict.items()
    for word in words
}

female_word_to_cluster = {
    word: int(cluster.replace("cluster_", ""))
    for cluster, words in female_keyword_dict.items()
    for word in words
}

# Step 1: Get all shared words
shared_words = set(male_word_to_cluster) & set(female_word_to_cluster)
print(f"🔁 Total overlapping words: {len(shared_words)}")

# Step 2: Words in same cluster (removed from both)
same_cluster = [
    word for word in shared_words
    if male_word_to_cluster[word] == female_word_to_cluster[word]
]
print(f"❌ Words with same cluster in both (removed from both): {len(same_cluster)}")

# Step 3: Words that stayed in male
male_preferred = [
    word for word in shared_words
    if male_word_to_cluster[word] < female_word_to_cluster[word]
]
print(f"✅ Words kept in male (lower cluster): {len(male_preferred)}")

# Step 4: Words that stayed in female
female_preferred = [
    word for word in shared_words
    if female_word_to_cluster[word] < male_word_to_cluster[word]
]
print(f"✅ Words kept in female (lower cluster): {len(female_preferred)}")




# Final male keywords:
# - Words where male has lower cluster
# - Words only in male
male_keywords = [
    word for word in male_word_to_cluster
    if (word not in female_word_to_cluster) or 
       (male_word_to_cluster[word] < female_word_to_cluster[word])
]

# Final female keywords:
# - Words where female has lower cluster
# - Words only in female
female_keywords = [
    word for word in female_word_to_cluster
    if (word not in male_word_to_cluster) or 
       (female_word_to_cluster[word] < male_word_to_cluster[word])
]

# Display results
print("✅ Final Male Keywords (including male-unique):")
print(male_keywords)

print("\n✅ Final Female Keywords (including female-unique):")
print(female_keywords)




def feature_engineer(df):
    
    # Determining action scores
    action_weights = {'visit': 1, 'search': 5, 'favorite': 10, 'basket': 10, 'order': 15}
    df_copy = df.copy()

    # Total number of user actions
    user_features_basic = df_copy.groupby('unique_id').agg({
        'user_action': 'count',
    }).reset_index()
    user_features_basic.columns = ['unique_id', 'total_actions']

    female = (df_copy['product_gender'] == 'Kadın').groupby(df_copy['unique_id']).sum().reset_index(name='female_product_count')
    male = (df_copy['product_gender'] == 'Erkek').groupby(df_copy['unique_id']).sum().reset_index(name='male_product_count')
    unisex = ((df_copy['product_gender'].isna()) | (df_copy['product_gender'] == 'Unisex')).groupby(df_copy['unique_id']).sum().reset_index(name='unisex_product_count')

    user_gender_features = female.merge(male, on='unique_id').merge(unisex, on='unique_id')

    user_features = user_features_basic.merge(user_gender_features, on='unique_id')

    action_counts = df_copy.pivot_table(index='unique_id', columns='user_action', aggfunc='size', fill_value=0).reset_index()
    action_columns = [col for col in action_counts.columns if col != 'unique_id']
    action_counts['total_actions'] = action_counts[action_columns].sum(axis=1)

    for col in ['favorite', 'order', 'basket']:
        if col not in action_counts.columns:
            action_counts[col] = 0

    # Action type ratios
    action_counts['favorite_ratio'] = action_counts['favorite'] / action_counts['total_actions'].replace(0, np.nan)
    action_counts['order_ratio'] = action_counts['order'] / action_counts['total_actions'].replace(0, np.nan)
    action_counts['basket_ratio'] = action_counts['basket'] / action_counts['total_actions'].replace(0, np.nan)

    action_counts[['favorite_ratio', 'order_ratio', 'basket_ratio']] = action_counts[[
        'favorite_ratio', 'order_ratio', 'basket_ratio'
    ]].fillna(0)

    user_features = user_features.merge(
        action_counts[['unique_id', 'favorite_ratio', 'order_ratio', 'basket_ratio']],
        on='unique_id',
        how='left'
    )
    user_features = user_features.fillna(0)

    total_known_products = (
            user_features['female_product_count'] +
            user_features['male_product_count'] +
            user_features['unisex_product_count']
    )

    # Product gender ratio and difference
    user_features['female_product_ratio'] = user_features['female_product_count'] / total_known_products.replace(0, np.nan)
    user_features['male_product_ratio'] = user_features['male_product_count'] / total_known_products.replace(0, np.nan)
    
    user_features['product_percent_difference'] = user_features['female_product_ratio'] - user_features['male_product_ratio']
    
    # Dropping intermediate columns
    user_features = user_features.drop(columns=['female_product_ratio', 'male_product_ratio'], errors='ignore')
    user_features = user_features.drop(columns=[
        'female_product_count',
        'male_product_count',
        'unisex_product_count',
        "total_actions"
    ])

    # Brand-based weighted gender scoring
    intent_actions = ['order', 'basket', 'favorite', 'search', 'visit']
    brand_score_df = df_copy[df_copy['user_action'].isin(intent_actions)].copy()

    brand_score_df['is_female_brand'] = brand_score_df['brand_name'].isin(female_brands).astype(int)
    brand_score_df['is_male_brand'] = brand_score_df['brand_name'].isin(male_brands).astype(int)
    
    brand_score_df['action_weight'] = brand_score_df['user_action'].map(action_weights)
    brand_score_df['female_brand_score_weighted'] = brand_score_df['is_female_brand'] * brand_score_df['action_weight']
    brand_score_df['male_brand_score_weighted'] = brand_score_df['is_male_brand'] * brand_score_df['action_weight']

    female_brand_score = brand_score_df.groupby('unique_id')['female_brand_score_weighted'].mean().reset_index(name='female_brand_score')
    male_brand_score = brand_score_df.groupby('unique_id')['male_brand_score_weighted'].mean().reset_index(name='male_brand_score')

    user_features = user_features.merge(female_brand_score, on='unique_id', how='left')
    user_features = user_features.merge(male_brand_score, on='unique_id', how='left')

    user_features[['female_brand_score', 'male_brand_score']] = user_features[['female_brand_score', 'male_brand_score']].fillna(0)

    # Difference of brand scores
    user_features['brand_score_difference'] = user_features['female_brand_score'] - user_features['male_brand_score']
    user_features = user_features.drop(columns=['female_brand_score', 'male_brand_score'], errors='ignore')

    # Keyword-based gender detection on product names
    df_copy['product_name_lower'] = df_copy['product_name'].str.lower().fillna("")
    df_copy['is_female_name_raw'] = df_copy['product_name_lower'].apply(lambda x: int(any(keyword in x for keyword in female_keywords)))
    df_copy['is_male_name_raw'] = df_copy['product_name_lower'].apply(lambda x: int(any(keyword in x for keyword in male_keywords)))

    filtered_actions = ['order', 'basket', 'favorite']

    df_copy['is_female_name'] = ((df_copy['user_action'].isin(filtered_actions)) & (df_copy['is_female_name_raw'] == 1)).astype(int)
    df_copy['is_male_name'] = ((df_copy['user_action'].isin(filtered_actions)) & (df_copy['is_male_name_raw'] == 1)).astype(int)

    df_copy = df_copy.drop(columns=['is_female_name_raw', 'is_male_name_raw'])
    user_features = user_features.drop(columns=['basket_ratio'])
    
    total_actions = df_copy.groupby('unique_id').size().reset_index(name='total_actions')

    user_features = user_features.merge(total_actions, on='unique_id', how='left')
    user_features['total_actions'] = user_features['total_actions'].fillna(0)

    # Unified gender tagging based on name, brand, or gender label
    df_copy['is_female_brand'] = df_copy['brand_name'].isin(female_brands).astype(int)
    df_copy['is_male_brand'] = df_copy['brand_name'].isin(male_brands).astype(int)

    df_copy['is_female_tagged'] = (
        (df_copy['product_gender'] == 'Kadın') |
        (df_copy['is_female_name'] == 1) |
        (df_copy['is_female_brand'] == 1)
    )

    df_copy['is_male_tagged'] = (
        (df_copy['product_gender'] == 'Erkek') |
        (df_copy['is_male_name'] == 1) |
        (df_copy['is_male_brand'] == 1)
    )

    # Category diversity per gender
    female_tagged = df_copy[df_copy['is_female_tagged']]
    male_tagged = df_copy[df_copy['is_male_tagged']]

    female_diversity = (
        female_tagged.groupby('unique_id')['Level3_Category_Name']
        .nunique()
        .reset_index(name='female_category_diversity')
    )

    male_diversity = (
        male_tagged.groupby('unique_id')['Level3_Category_Name']
        .nunique()
        .reset_index(name='male_category_diversity')
    )

    total_diversity = (
        df_copy.groupby('unique_id')['Level3_Category_Name']
        .nunique()
        .reset_index(name='total_category_diversity')
    )

    diversity_df = total_diversity.merge(female_diversity, on='unique_id', how='left')
    diversity_df = diversity_df.merge(male_diversity, on='unique_id', how='left')
    diversity_df[['female_category_diversity', 'male_category_diversity']] = diversity_df[[
        'female_category_diversity', 'male_category_diversity'
    ]].fillna(0)

    # Normalized difference in category diversity
    diversity_df['normalized_category_diversity_diff'] = (
        (diversity_df['female_category_diversity'] / diversity_df['total_category_diversity'].replace(0, np.nan)) -
        (diversity_df['male_category_diversity'] / diversity_df['total_category_diversity'].replace(0, np.nan))
    ).fillna(0)

    user_features = user_features.merge(
        diversity_df[['unique_id', 'normalized_category_diversity_diff']],
        on='unique_id', how='left'
    )
    user_features['normalized_category_diversity_diff'] = user_features['normalized_category_diversity_diff'].fillna(0)

    
    return user_features


data_ready = feature_engineer(train_data)
user_labels = train_data[['unique_id', 'gender']].drop_duplicates()
data_ready = pd.merge(data_ready, user_labels, on='unique_id')

le = LabelEncoder()
y = le.fit_transform(data_ready['gender'])
X = data_ready.drop(columns=['unique_id', 'gender'])
feature_columns = X.columns.tolist() 

y_0 = sum(y == 0)
y_1 = sum(y == 1)
scale_pos_weight = y_0 / y_1
# We extract the parameters below and then manually added to our model
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

sampler = TPESampler(seed=42)
xgb_study = optuna.create_study(direction='maximize', sampler=sampler)

# Objective function to optimize AUC
def xgb_objective(trial):
    params = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
        'random_state': 42

    }
    # Evaluating average AUC over 5 folds
    model = xgb.XGBClassifier(**params, use_label_encoder=False)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    return scores.mean()


[I 2025-05-25 17:01:20,297] Trial 7 finished with value: 0.8896979833451872 and parameters: {'max_depth': 3, 'learning_rate': 0.0668350301015521, 'n_estimators': 118, 'subsample': 0.6626651653816322, 'colsample_bytree': 0.6943386448447411, 'reg_alpha': 0.2713490317738959, 'reg_lambda': 0.8287375091519293, 'min_child_weight': 4.210779940242304}. Best is trial 7 with value: 0.8896979833451872.
# Retrieving best parameters
xgb_study.optimize(xgb_objective, n_trials=30)
best_xgb_params = xgb_study.best_params
best_xgb_params['scale_pos_weight'] = scale_pos_weight
xgb_model = xgb.XGBClassifier(**best_xgb_params, use_label_encoder=False, eval_metric='auc', random_state=42)"""
xgb_model = xgb.XGBClassifier(max_depth= 3, scale_pos_weight=scale_pos_weight , learning_rate= 0.0668350301015521, n_estimators= 118, subsample=0.6626651653816322, colsample_bytree=0.6943386448447411, reg_alpha=0.2713490317738959, reg_lambda=0.8287375091519293, min_child_weight= 4.210779940242304, use_label_encoder=False, eval_metric='auc', random_state=42, n_jobs=1)
# Preparing test data
test_data = pd.read_csv('./IE425_Spring25_test_data.csv')
test_ready = feature_engineer(test_data)

le = LabelEncoder()
y = le.fit_transform(data_ready['gender'])
X = data_ready.drop(columns=['unique_id', 'gender'])
feature_columns = X.columns.tolist()

xgb_model.fit(X,y)

for col in feature_columns:
    if col not in test_ready.columns:
        test_ready[col] = 0
X_test = test_ready[feature_columns]

probs = xgb_model.predict_proba(X_test)
female_index = list(le.classes_).index('F')
prob_female = probs[:, female_index]
y_pred_test = xgb_model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred_test)

# Preparing submission file
submission = pd.DataFrame({
    'unique_id': test_ready['unique_id'],
    'probability_female': prob_female,
    'gender': y_pred_labels
})

submission.to_csv("test_prediction.csv", index=False)


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Generate learning curve
train_sizes, train_scores, val_scores = learning_curve(
    xgb_model,
    X,
    y,
    cv=5,
    scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1,
    shuffle=True,
    random_state=42
)

# Calculate means and standard deviations
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plotting the curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training AUC', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

plt.plot(train_sizes, val_mean, 'o-', label='Validation AUC', color='orange')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')

plt.title('Learning Curve for XGBoost Model')
plt.xlabel('Training Set Size')
plt.ylabel('ROC AUC Score')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()