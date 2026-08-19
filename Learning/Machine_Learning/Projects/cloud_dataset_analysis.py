import os
from pathlib import Path

import pandas as pd

file_path = Path(os.environ.get("CLOUD_TWEETS_DATASET", "Data/Raw/cloud_tweets.csv.gz"))
if not file_path.exists():
	raise FileNotFoundError(f"Set CLOUD_TWEETS_DATASET to the downloaded dataset path: {file_path}")

df = pd.read_csv(file_path, compression="gzip")

print(df.columns.tolist())
print(df.head())
print(df.info())