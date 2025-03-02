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

from datasets import Dataset

CITATION_REGEX = r" ?(\[\d+(?:-\d+|(?:, ?\d+(-\d+)?)*)+\]|<([A-Z]+:[a-zA-Z0-9._:/-]*)>) ?"


def generate_citing_sentences_dataset(data_dir: str, output_dir: str) -> str:
    # Glob all .txt files in the hierarchy
    files = []

    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                files.append(os.path.join(root, filename))

    print(f"Found {len(files)} files")

    df = pd.DataFrame(map(lambda sentence: [re.sub(CITATION_REGEX,
                                                   "",
                                                   sentence),
                                            bool(re.search(CITATION_REGEX,
                                                           sentence))
                                            ],
                          [sentences for file in files for sentences in
                           open(file, "r", encoding="utf-8").read().split("\n============\n")]),
                      columns=["sentence", "citing"])

    df.to_csv(os.path.join(output_dir, "citing_sentences.csv"), index=False)

    return os.path.join(output_dir, "citing_sentences.csv")


def generate_hf_dataset(path: str, output_dir: str) -> None:
    ds = Dataset.from_csv(path)
    ds.save_to_disk(output_dir + "/citing_sentences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with the raw data files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the generated dataset")
    args = parser.parse_args()

    # Generation done in two steps: raw data -> csv -> HuggingFace dataset
    csv_path = generate_citing_sentences_dataset(args.data_dir, args.output_dir)
    generate_hf_dataset(csv_path, args.output_dir)
