import pandas as pd
import os

DATA_PATH = 'data/news.csv'

try:
    print(f"Reading {DATA_PATH}...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Line 18: {lines[17]}") # 0-indexed, so line 18 is index 17
        print(f"Line 19: {lines[18]}")

    df = pd.read_csv(DATA_PATH)
    print("Successfully read CSV.")
    print(df.head())
    print(df.shape)
except Exception as e:
    print(f"Error: {e}")
