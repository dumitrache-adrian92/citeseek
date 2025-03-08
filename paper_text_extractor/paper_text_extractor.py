"""
Utility for extracting text from PDF files with additional features to process
scientific papers by optionally omitting sections such as abstracts, references,
and citation markers.

It allows command-line usage to efficiently extract, clean, and preprocess text 
content from scientific documents in bulk or individually.
"""

import re

from pathlib import Path
from typing import IO
from pdfminer.high_level import extract_text


def get_paper_text(path: str | Path | IO,
                   remove_references: bool = False,
                   remove_abstract: bool = False,
                   remove_reference_markers: bool = False) -> str:
    """
    Extracts text content from a PDF file, with options to remove specific sections
    or markers.

    This function reads the content of a PDF document specified by the given file
    path and returns the extracted text. It can optionally exclude references,
    abstracts, or reference markers if specified.

    Args:
        path (str | Path | IO): The file path to the PDF document.
        remove_references (bool): A flag indicating whether to remove the References
            section from the extracted text. Default is False.
        remove_abstract (bool): A flag indicating whether to remove the Abstract
            section from the extracted text. Default is False.
        remove_reference_markers (bool): A flag indicating whether to remove
            reference markers (e.g., [1], [1, 2], [1-3]) from the extracted text.
            Default is False.

    Returns:
        str: The processed text extracted from the PDF document.
    """
    pdf_text = extract_text(path)

    if remove_references:
        pdf_text = pdf_text.split('References', 1)[0]

    if remove_abstract:
        pdf_text = pdf_text.split('Abstract', 1)[1]

    if remove_reference_markers:
        pdf_text = re.sub(r'\[.*]', '', pdf_text)
        pdf_text = re.sub(r'\[\s*\d+\s*]', '', pdf_text)
        pdf_text = re.sub(r'\[\s*\d+(,\s*\d+)*\s*]', '', pdf_text)
        pdf_text = re.sub(r'\[\s*\d+\s*-\s*\d+\s*]', '', pdf_text)

    return pdf_text
