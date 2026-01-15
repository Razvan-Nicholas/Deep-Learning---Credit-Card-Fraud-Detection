from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(X_train, y_train, random_state=42):
    # Uses balanced class weights to handle imbalance as mentioned in notebook
    lr = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=random_state)
    lr.fit(X_train, y_train)
    return lr

def train_random_forest(X_train, y_train, random_state=42):
    # Captures non-linear interactions [cite: 93]
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=random_state)
    rf.fit(X_train, y_train)
    return rf
