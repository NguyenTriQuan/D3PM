import json
import datasets
import os
import gdown
import requests
import tarfile
import zipfile
import urllib.request
import dload


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
        output_file = 'conditional_data/description2code/description2code.zip' 
        # url = 'https://drive.google.com/drive/u/0/folders/1CNmq-ANr_FNw2o7Y-zSrFvQsSKX0CvGQ'
        url = 'https://drive.google.com/file/d/1UEqP1GpaIS2cSoVwKDezR2HqE3RUv-MT'
        gdown.download(url, output_file, quiet=False)
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            # Extract all the contents to the destination folder
            zip_ref.extractall('conditional_data/description2code')

        print("File extracted successfully.")
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