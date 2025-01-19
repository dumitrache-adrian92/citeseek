"""
Utility for extracting text from PDF files with additional features to process
scientific papers by optionally omitting sections such as abstracts, references,
and citation markers.

It allows command-line usage to efficiently extract, clean, and preprocess text 
content from scientific documents in bulk or individually.
"""

import pypdf
import re


def get_paper_text(path, remove_references=False, remove_abstract=False, remove_reference_markers=False):
    """
    Extracts text content from a PDF file, with options to remove specific sections 
    or markers. This function reads the content of a PDF document specified by the 
    given file path and returns the extracted text. It can optionally exclude 
    references, abstracts, or reference markers if specified.

    :param path: The file path to the PDF document.
    :type path: str
    :param remove_references: A flag indicating whether to remove the References 
        section from the extracted text. Default is False.
    :type remove_references: bool
    :param remove_abstract: A flag indicating whether to remove the Abstract section 
        from the extracted text. Default is False.
    :type remove_abstract: bool
    :param remove_reference_markers: A flag indicating whether to remove reference 
        markers (e.g., [1], [1, 2], [1-3]) from the extracted text. Default is False.
    :type remove_reference_markers: bool
    :return: The processed text extracted from the PDF document.
    :rtype: str
    """
    py_pdf = pypdf.PdfReader(path)

    pdf_text = ''.join([page.extract_text() for page in py_pdf.pages])

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
