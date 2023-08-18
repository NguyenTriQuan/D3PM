import json
import datasets
import os
import gdown

class QT(datasets.GeneratorBasedBuilder):

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
        data_dir = 'conditional_data/Q-T' 
        url = 'https://drive.google.com/drive/folders/122YK0IElSnGZbPMigXrduTVL1geB4wEW'
        gdown.download_folder(url, output=data_dir, quiet=True, use_cookies=False)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, "QT/test.jsonl")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir, "QT/val.jsonl")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "QT/train.jsonl")}
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