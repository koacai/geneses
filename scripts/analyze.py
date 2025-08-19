from pathlib import Path

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    test_dir = Path("test_output")

    without_ref_paths = list(test_dir.glob("**/without_ref.csv"))
    print(f"Found {len(without_ref_paths)} without_ref.csv files")

    all_dfs = []
    for csv_path in tqdm(without_ref_paths, desc="Loading CSV files"):
        df = pd.read_csv(csv_path)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    metrics = ["dnsmos", "nisqa", "utmos"]
    grouped_averages = combined_df.groupby("key")[metrics].mean()

    print("\nAverage scores grouped by key:")
    print("-" * 50)
    print(grouped_averages)

    with_ref_paths = list(test_dir.glob("**/with_ref.csv"))
    print(f"Found {len(with_ref_paths)} with_ref.csv files")

    all_dfs = []
    for csv_path in tqdm(with_ref_paths, desc="Loading CSV files"):
        df = pd.read_csv(csv_path)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    metrics = ["pesq", "estoi", "sdr", "mcd", "lsd", "speech_bert_score"]
    grouped_averages = combined_df.groupby("key")[metrics].mean()

    print("\nAverage scores grouped by key:")
    print("-" * 50)
    print(grouped_averages)
