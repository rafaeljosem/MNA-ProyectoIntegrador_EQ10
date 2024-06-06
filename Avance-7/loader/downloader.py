"""
Downloader functions
"""
import os
import re
from zipfile import ZipFile

import requests
from requests.exceptions import RequestException


def extract_filename(r: requests.Response) -> str:
    """
    Extracts the filename from the response
    """
    if "Content-Disposition" in r.headers.keys():
        return re.findall("filename=(.+)",
                          r.headers["Content-Disposition"])[0]

    return r.url.split("/")[-1]


def download_files(urls: list, directory: str):
    """
    Downloads a file
    """
    filenames = []
    if not os.path.exists(directory):
        os.mkdir(directory)
    for url in urls:
        try:
            with requests.get(url, timeout=60) as r:
                filename = extract_filename(r)
                filenames.append(filename)
                fullpath = os.path.join(directory, filename)
                create_file(fullpath, r.content)

                if is_zip(filename):
                    unzip(filename, directory, directory)
                    print(fullpath)
                    os.remove(fullpath)

        except RequestException as e:
            print(e)
    return filenames[0] if len(filenames) == 1 else filenames


def create_file(full_path: str, content):
    """
    Creates the downloaded file
    """

    with open(full_path, mode="wb") as f:
        f.write(content)


def unzip(filename: str, source_path: str, dest_path: str):
    """
    Unzips a file
    """
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    with ZipFile(f'{source_path}/{filename}', 'r') as z_object:
        z_object.extractall(path=dest_path)


def is_zip(file: str):
    """
    Checks if file is a zip file
    """
    if file.lower().endswith('.zip'):
        return True
    return False
