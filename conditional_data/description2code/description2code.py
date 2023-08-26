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
    src_data = []
    trg_data = []
    for prob in os.listdir(data_dir):
        src_file = data_dir + '/' + prob + '/' + 'description/description.txt'
        with open(trg_dir + '/' + solution, 'r') as f:
            src_data.append(f.read().strip())

        trg_dir = data_dir + '/' + prob + '/' + 'solutions_c++'
        for solution in os.listdir(trg_dir):
            with open(trg_dir + '/' + solution, 'r') as f:
                trg_data.append(f.read().strip())
    print(len(src_data), len(trg_data))
    print(src_data[0])
    print(trg_data[0])



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