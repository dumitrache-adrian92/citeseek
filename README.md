# CiteSeek

CiteSeek is an end-to-end pipeline for identifying missing citations in
scientific papers in an explainable manner. It works by identifying passages
which require citations (and do not already cite other works) and using them to
retrieve a set of candidate papers from a database of paper titles and
abstracts based on their embeddings.

This means that CiteSeek's output includes what papers need to be cited, as
well as which passages should cite them, making it quite valuable in academic
integrity checks and real time document editing. It excels at identifying
common mistakes such as uncited named entities or uncited statistics and does
decently well in other scenarios, though its recommendations become less
precise.

This repo was part of my bachelor's diploma project at POLITEHNICA University
of Bucharest. If you're interested in a detailed overview of how all of this
works, my entire thesis can be found in the `thesis/` folder.

## Usage

You will at the very least need to have Docker set up. Conda is also very
useful to replicate the environment that I used. Note that you need to fill
up the database with your own papers using the `index_papers` method,
[S2ORC](https://github.com/allenai/s2orc) is a great place to start.

```bash
conda env create -f environment.yaml # if using conda
pip install -r requirements.txt # if you want to use pip instead

docker compose up # start the database

python -m missing_citation_retriever <path_to_pdf>
```

## Collaboration

Hit me up on any platform if you're interested in this task (or any other task
on scholarly documents) and want some guidance.  
