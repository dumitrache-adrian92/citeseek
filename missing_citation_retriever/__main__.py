from pathlib import Path
from argparse import ArgumentParser
from .missing_citation_retriever import MissingCitationRetriever


if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser(
        description='This script extracts and processes text from scientific papers in '
                    'PDF format. It includes options to remove abstracts, references, '
                    'and citation markers for easier text analysis.')
    parser.add_argument('path', type=Path, help='Path to the PDF file')
    args = parser.parse_args()

    CLASSIFIER_PATH = "C:\\Users\\Adrian\\Documents\\models\\citing_sentence_classifier"

    retriever: MissingCitationRetriever = MissingCitationRetriever()

    print(retriever.check_paper(args.path))
