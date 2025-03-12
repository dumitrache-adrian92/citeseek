"""
Script to generate a HuggingFace dataset of sentences from a corpus of sentences extracted from scientific
papers classified as either citing or non-citing sentences.

The raw data is formed of text files for each paper with sentences separated by `\n============\n`.

The sentences can be classified by the presence of a citation in the sentence, e.g.: `[1]`,
`<GC:and.downey.fellows.vardy.whittle>`.

The script will generate a dataset with the following columns:
- `sentence`: the sentence text
- `citing`: whether the sentence contains a citation
"""

import argparse
import os
import pandas as pd
import re

from sklearn.model_selection import train_test_split

TEST_PQ = "citing_test.parquet"
VAL_PQ = "citing_val.parquet"
TRAIN_PQ = "citing_train.parquet"

CITATION_REGEX = r" ?(\[\d+(?:-\d+|(?:, ?\d+(-\d+)?)*)+\]|<([A-Z]+:[a-zA-Z0-9._:/-]*)>) ?"


def generate_citing_sentences_dataset(data_dir: str, output_dir: str):
    # Glob all .txt files in the hierarchy
    files = []

    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                files.append(os.path.join(root, filename))

    print(f"Found {len(files)} files")

    # Construct the dataset in memory (requires 3 GB of RAM)
    df = pd.DataFrame(
        map(
            lambda sentence: [
                re.sub(CITATION_REGEX, "", sentence),
                bool(re.search(CITATION_REGEX, sentence)),
            ],
            [
                sentences
                for file in files
                for sentences in open(file, "r", encoding="utf-8")
                .read()
                .split("\n============\n")
            ],
        ),
        columns=["sentence", "citing"],
    )

    # split the dataset into training, validation and test sets (80/10/10)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_test, df_val = train_test_split(df_test, test_size=0.5, random_state=42)

    # save the datasets to disk
    df_train.to_parquet(
        os.path.join(output_dir, TRAIN_PQ),
        engine="pyarrow",
        compression="snappy",
    )
    df_val.to_parquet(
        os.path.join(output_dir, VAL_PQ),
        engine="pyarrow",
        compression="snappy",
    )
    df_test.to_parquet(
        os.path.join(output_dir, TEST_PQ),
        engine="pyarrow",
        compression="snappy",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with the raw data files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the generated dataset")
    args = parser.parse_args()

    generate_citing_sentences_dataset(args.data_dir, args.output_dir)
