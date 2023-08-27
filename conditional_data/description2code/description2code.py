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
    num_description = 0
    avg_solutions_per_description = 0
    avg_testcases_per_description = 0
    with open('conditional_data/description2code/description2code_current/codeforces/data.jsonl', 'w') as jsonl_file:
        for prob in os.listdir(data_dir):
            src_file = data_dir + '/' + prob + '/' + 'description/description.txt'
            src_exist = os.path.isfile(src_file)
            samples_dir = data_dir + '/' + prob + '/' + 'samples'
            samples_exist = os.path.isfile(samples_dir)
            trg_dir = data_dir + '/' + prob + '/' + 'solutions_python'
            if os.path.isdir(trg_dir):
                trg_count = len(os.listdir(trg_dir))

            if src_exist and trg_count > 0 and samples_exist:
                testcases = []
                for inp in os.listdir(samples_dir):
                    if 'input' in inp:
                        with open(samples_dir + '/' + inp, 'r') as f:
                            inp_test = f.read()
                        if len(inp_test) > 0:
                            with open(samples_dir + '/' + inp[:-9] + 'output.txt', 'r') as f:
                                out_test = f.read()
                            if len(out_test) > 0:
                                testcases.append({'input':inp_test, 'output':out_test})

                if len(testcases) == 0: continue

                with open(src_file, 'r') as f:
                    description = f.read().strip()
                
                if len(description) == 0: continue

                for solution in os.listdir(trg_dir):
                    with open(trg_dir + '/' + solution, 'r') as f:
                        code = f.read().strip()

                    if len(code) == 0: continue
                    avg_solutions_per_description += 1

                    data = {'src':description, 'trg':code, 'test': testcases}
                    json.dump(data, jsonl_file)
                    jsonl_file.write('\n')

                print(prob, 'successed')
                num_description += 1
                avg_testcases_per_description += len(testcases)
        
        print('\n### Dataset Info: ###\n')
        print('\n### Description: ###\n')
        print('Total description', num_description)
        print(data['src'])
        print('\n### Solution: ###\n')
        print('Total solution', avg_solutions_per_description)
        print('Avg solution per description', avg_solutions_per_description / num_description)
        print(data['trg'])
        print('\n### Test Cases: ###\n')
        print('Avg Test cases per description', avg_testcases_per_description / num_description)
        print(data['test'])


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
        data_dir = 'conditional_data/description2code/description2code_current/codeforces'
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