import pandas as pd

print("Loading unprocessed data ...")
unprocessed_data = pd.read_csv("data/test.csv")

print("Preprocessing ...")
# PREPROCESS DATA HERE
processed_data = unprocessed_data

processed_data.to_csv("data/preprocessed_data.csv", index=False)

print("Preprocessing complete!")