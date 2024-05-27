# legal_document_loader.py

"""
Module: legal_document_loader.py
Description: Module for loading legal documents from various sources.
"""

import os
import re
from typing import Dict, List

import pandas as pd
from encoder.utf8_encoder import convert_to_utf8
from loader.downloader import download_files
from logger.logger import Logger


def remove_new_lines(lines: list) -> list:
    """
    This method removes every new line
    from a string
    """
    cleaned_text = []
    for line in lines:
        if not isinstance(line, str):
            continue
        line = line.strip(' ')
        if line == '':
            continue
        cleaned_text.append(line)
    return cleaned_text


def clean_text(doc):
    """
    Function that removes more that two subsequent line breaks

    :param doc: The text content of the legal document.
    """
    if isinstance(doc, (str, bytes)):
        # Remove all subsequent line breaks greater than 2
        cleaned_text = re.sub('\n{1,} | \n', '\n', doc)
        cleaned_text = re.sub('\n{2,}', '\n\n', cleaned_text)

        return cleaned_text
    return ""


class LegalDocumentLoader:
    """
    Class for loading legal documents from various sources.
    """
    logger = Logger()

    @staticmethod
    def load_from_url(url: str) -> List[Dict[str, str]]:
        """
        Load legal documents from a URL.

        :param url: The URL containing the ZIP file with the legal documents.
        :return: List of dictionaries, where each dictionary represents a
        legal document with keys 'Title', 'Filename', and 'Text'.
        """

        # Extract the contents of the ZIP file
        dest_dir = "legal_documents"
        download_files([url], dest_dir)
        legal_documents = []

        # Iterate over the extracted files and read their content
        for path, _, files in os.walk(dest_dir):
            for file in files:
                file_path = os.path.join(path, file)
                if file.endswith(".txt"):
                    legal_documents.extend(
                        LegalDocumentLoader.load_from_txt(file_path, file))
                elif file.endswith(".csv"):
                    legal_documents.extend(
                        LegalDocumentLoader.load_from_csv(file_path))

                else:
                    legal_documents.extend(
                        LegalDocumentLoader.load_from_parquet(file_path))

        return legal_documents

    @ staticmethod
    def load_from_txt(file_path: str, file) -> List[Dict[str, str]]:
        """
        Loads content from a txt file
        """
        legal_documents = []
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read().splitlines()
                cleaned_text = remove_new_lines(text)

                if len(cleaned_text) != 0:
                    # Extract the title from the first line
                    title = cleaned_text[0].strip()
                    title = convert_to_utf8(
                        title.encode('latin-1'))

                    # Join the remaining lines as the content
                    content = "\n".join(cleaned_text[1:])
                    content = convert_to_utf8(
                        content.encode('latin-1'))

                    legal_documents.append({
                        "Title": title,
                        "Filename": file,
                        "Text": content
                    })
        except UnicodeDecodeError:
            print(
                f"**Skipping file {file_path} due to encoding issues.**\n")

        return legal_documents

    @ staticmethod
    def load_from_csv(file_path: str) -> List[str]:
        """
        Load legal documents from a CSV file.

        :param file_path: The path to the
        CSV file containing the legal documents.

        :return: List of strings, where each string
        represents the text content of a legal document.
        """
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path, index_col=0)
        return LegalDocumentLoader.extract_legal_documents_from_dataframe(df)

    @staticmethod
    def load_from_parquet(file_path: str) -> List[str]:
        """
        Load legal documents from a parquet file.

        :param file_path: The path to the
        parquet file containing the legal documents.

        :return: List of strings, where each string
        represents the text content of a legal document.
        """
        # Read the parquet file into a Pandas DataFrame
        df = pd.read_parquet(file_path)
        return LegalDocumentLoader.extract_legal_documents_from_dataframe(df)

    @staticmethod
    def extract_legal_documents_from_dataframe(df: pd.DataFrame) -> list[str]:
        """
        Reads the dataframe and extract the legal
        text to be used
        """
        name_list = df.index.to_list()

        # Check if the 'Text' column exists in the DataFrame
        if 'Text' not in df.columns:
            raise ValueError(
                "The 'Text' column is missing in the dataframe.")

        # Apply the clean_text function to the 'Text' column and
        # create a new 'Clean Text' column
        df['Clean Text'] = df['Text'].apply(
            lambda x: clean_text(
                x) if isinstance(x, (str, bytes)) else "")

        # Extract the text content from the 'Clean Text' column
        # and convert it to a list of strings
        text_list = df['Clean Text'].tolist()

        # Remove any None or empty values from the list
        text_list = [doc for doc in text_list if doc and isinstance(doc, str)]

        # Initialize an empty list called 'legal_documents'
        legal_documents = []

        # Iterate through the elements of 'name_list'
        # and 'text' simultaneously using zip
        for name_item, text_item in zip(name_list, text_list):
            # Create a dictionary with keys 'name' and 'text',
            # and assign corresponding values
            data_dict = {'Title': name_item.title(), 'Text': text_item}

            # Append the created dictionary to the 'data' list
            legal_documents.append(data_dict)

        return legal_documents
