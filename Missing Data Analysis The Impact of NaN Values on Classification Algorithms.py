import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os

root = tk.Tk()
root.withdraw()  
file_path = filedialog.askopenfilename()

df = pd.read_csv(file_path)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifiers = {
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier()
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"{name} Confusion Matrix:\n{cm}")

nan_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[nan_indices, df.columns[:-1]] = np.nan

file_dir, file_name = os.path.split(file_path)
nan_file_name = os.path.join(file_dir, "withNaN_" + file_name)
df.to_csv(nan_file_name, index=False)

print(f"NaN değerleri içeren veri seti '{nan_file_name}' olarak kaydedildi.")

df.fillna(df.mean(), inplace=True)

nan_filled_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Save the file with NaN values replaced")
df.to_csv(nan_filled_path, index=False)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"{name} (After NaN replacement) Confusion Matrix:\n{cm}")
