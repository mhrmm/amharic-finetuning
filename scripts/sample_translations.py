import json
import random

EXP_DIR = "experiments/en-am-nllb-1.3B-v0/"


source_file = "/mnt/storage/data/am-en/afridoc/validation.health.en"
references_file = f"{EXP_DIR}/references.json"
translations_file = f"{EXP_DIR}/translations.json"

with open(translations_file) as reader:
    translations = json.load(reader)["eng_Latn->amh_Ethi"]

with open(references_file) as reader:
    references = json.load(reader)["eng_Latn->amh_Ethi"]

with open(source_file) as reader:
    sources = []
    for line in reader:
        sources.append(line)

line_nums = random.sample(list(range(len(sources))), 50)
for line_num in line_nums:
    print("---")
    print("source:")
    print(sources[line_num])
    print("translation:")
    print(translations[line_num])
