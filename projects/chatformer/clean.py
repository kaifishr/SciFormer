"""Cleans transcript from Lex Fridman episodes.

Script removes time stamps and merges all transcript into a ~60MB file.

Transcripts can be found here: https://karpathy.ai/lexicap/
"""
import re
import os
import zipfile
import pathlib

import torchtext


def run_clean():

    # Create folder for data.
    data_dir = "data/lexicap/"
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Download data if not already done.
    dataset_url = "https://karpathy.ai/lexicap/data.zip"
    torchtext.utils.download_from_url(url=dataset_url, root=data_dir)

    # Define regular expression pattern to remove time stamps.
    pattern = r"(\s)?(\d{1,2}:)?\d{2}:\d{2}.\d{3} --> (\d{1,2}:)?\d{2}:\d{2}.\d{3}"
    # Compile the regular expression
    regex = re.compile(pattern)

    transcripts = []

    cwd = os.getcwd()
    with zipfile.ZipFile(cwd + "/" + data_dir + "data.zip", mode="r") as zip:
        for name in zip.namelist():
            # There are "small" and "large" files
            # for every transcript. Here we go with "large".
            if name.endswith("large.vtt"):
                with zip.open(name, mode="r") as file:
                    # Skip the header.
                    file.readline()
                    # Encode data.
                    data = str(file.read(), encoding="utf-8")
                    # Remove new lines. 
                    data = " ".join(data.split())
                    # Remove time stamps with pattern defined above.
                    data = regex.sub("", data)
                    transcripts.append(data)

    transcripts = " ".join(transcripts)
    with open("output.txt", "w") as out:
        out.write(transcripts)

if __name__ == "__main__":
    run_clean()