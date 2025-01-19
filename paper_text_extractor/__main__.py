import argparse
from .paper_text_extractor import get_paper_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script extracts and processes text from scientific papers in '
                                                 'PDF format. It includes options to remove abstracts, references, '
                                                 'and citation markers for easier text analysis.')
    parser.add_argument('path', type=str, help='Path to the PDF file')
    parser.add_argument('--remove-references', action='store_true', help='Remove references')
    parser.add_argument('--remove-abstract', action='store_true', help='Remove abstract')
    parser.add_argument('--remove-reference-markers', action='store_true', help='Remove reference markers')
    args = parser.parse_args()

    text = get_paper_text(args.path, args.remove_references, args.remove_abstract, args.remove_reference_markers)
    print(text.encode('ascii', 'ignore').decode())
