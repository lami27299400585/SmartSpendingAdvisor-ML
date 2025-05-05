import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import os

def load_file(filepath):
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format.")

def give_budget_tips(df):
    tips = []
    total = df["amount"].sum()
    category_totals = df.groupby("Predicted Category")["amount"].sum()

    for cat, amt in category_totals.items():
        pct = (amt / total) * 100
        if pct > 20:
            tips.append(f"⚠️ High spending in '{cat}' ({pct:.1f}%)")
        elif pct < 5:
            tips.append(f"✅ Low spending in '{cat}' ({pct:.1f}%)")
    return tips

def cluster(df, vec, k=3):
    vecs = vec.transform(df["description"])
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    df["Cluster"] = km.fit_predict(vecs)
    return df

def main():
    # Full path to your CSV file in Downloads
    downloads_path = os.path.expanduser("~\\Downloads\\sample_expenses.csv")
    print(f"🔁 Loading file from: {downloads_path}")
    df = load_file(downloads_path)
    df.dropna(subset=["description", "amount", "category"], inplace=True)

    X = df["description"]
    y = df["category"]

    vec = TfidfVectorizer(stop_words="english")
    X_vec = vec.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, model.predict(X_test)))

    df["Predicted Category"] = model.predict(X_vec)

    print("\n🔎 Sample Predictions:")
    print(df[["description", "amount", "Predicted Category"]].head())

    print("\n💡 Budgeting Tips:")
    for tip in give_budget_tips(df):
        print(tip)

    df = cluster(df, vec)
    print("\n🧩 Sample Clusters:")
    print(df[["description", "Cluster"]].head())

if __name__ == "__main__":
    main()

