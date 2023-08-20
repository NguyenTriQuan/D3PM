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
        data_dir = 'conditional_data/description2code' 
        url = 'https://www.dropbox.com/s/zwj6u4caehf54s0/description2code_current.zip'
        dload.save_unzip(url, data_dir)
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