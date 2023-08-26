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
    data = []
    with open('conditional_data/description2code/description2code_current/codeforces/data.jsonl', 'w') as jsonl_file:
        for prob in os.listdir(data_dir):
            print(prob)
            src_file = data_dir + '/' + prob + '/' + 'description/description.txt'
            src_exist = os.path.isfile(src_file)
            trg_dir = data_dir + '/' + prob + '/' + 'solutions_python'
            if os.path.isdir(trg_dir):
                trg_count = len(os.listdir(trg_dir))

            if src_exist and trg_count > 0:
                with open(src_file, 'r') as f:
                    description = f.read().strip()

                for solution in os.listdir(trg_dir):
                    with open(trg_dir + '/' + solution, 'r') as f:
                        code = f.read().strip()
                        json.dump({'src':description, 'trg':code}, jsonl_file)
                        jsonl_file.write('\n')


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
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, "data.jsonl")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir, "data.jsonl")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "data.jsonl")}
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