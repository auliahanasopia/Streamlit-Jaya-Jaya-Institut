import pandas as pd
import sqlite3

df = pd.read_csv("c:/ASAH/submission_JayaJayaInstitut/submission_institut/data.csv", sep=";")

# Kolom bantu
df["Dropout"] = (df["Status"] == "Dropout").astype(int)
df["AgeGroup"] = pd.cut(df["Age_at_enrollment"],
    bins=[17,20,23,26,30,70],
    labels=["18-20","21-23","24-26","27-30","30+"])

conn = sqlite3.connect("jaya_jaya_institut.db")
df.to_sql("students", conn, if_exists="replace", index=False)
conn.close()
print("Database berhasil dibuat!")