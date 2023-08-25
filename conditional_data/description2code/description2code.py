import json
import datasets
import os
import gdown
import requests
import tarfile
import zipfile
import urllib.request
import dload

def download_and_unzip():
    data_dir = 'conditional_data/description2code' 
    url = 'https://drive.google.com/drive/folders/1CNmq-ANr_FNw2o7Y-zSrFvQsSKX0CvGQ'
    gdown.download_folder(url, output=data_dir, quiet=True, use_cookies=False)
    with zipfile.ZipFile('conditional_data/description2code/description2code_current.zip', 'r') as zip_ref:
        # Extract all the contents to the destination folder
        zip_ref.extractall('conditional_data/description2code')

    print("File extracted successfully.")

def extract_data():
    data_dir = 'conditional_data/description2code/description2code_current/codeforces'
    for file in os.listdir(data_dir):
        print(file)



class D2C(datasets.GeneratorBasedBuilder):

    def _info(self):
        features = datasets.Features(
            {
                "src": datasets.Value("string"),
                "trg": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        data_dir = 'conditional_data/description2code' 
        # download_and_unzip()
        extract_data()
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, "test.jsonl")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir, "valid.jsonl")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "train.jsonl")}
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                line = json.loads(line.strip())
                yield idx, {
                    "src": line["src"],
                    "trg": line["trg"],
                }