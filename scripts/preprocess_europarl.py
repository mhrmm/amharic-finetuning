"""
Europarl v7 can be found here: https://www.statmt.org/europarl/.

It has twenty X<>E parallel corpora from European parliament proceedings. 
From these raw files, this script creates a directory containing parallel files
corresponding to each language, i.e.:

bulgarian.txt
czech.txt
danish.txt
english.txt
etc.

- The files will have the same number of lines.
- Line k of each file corresponds to meaning of line k in english.txt.
- There are no duplicate lines in english.txt.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import random

parser = argparse.ArgumentParser(description="Preprocessing script for raw Europarl files.")
parser.add_argument("--data_dir", type=str, required=True, help="Directory with raw Europarl files.")
parser.add_argument("--out_dir", type=str, required=True, help="Directory for storing the preprocessed data.")
parser.add_argument("--min_length", type=int, default=10, help="Minimum length of (English) sentences to keep.")
parser.add_argument("--seed", type=int, default=42, help="Random seed (for shuffling).")
parser.add_argument("--num_dev", type=int, default=5000, help="Number of sentences to reserve for the dev set.")
parser.add_argument("--num_test", type=int, default=5000, help="Number of sentences to reserve for the test set.")
args = parser.parse_args()
data_dir = Path(args.data_dir)        
out_dir = Path(args.out_dir)  
out_dir.mkdir(exist_ok=True)
language_codes = {
    "bg": "bulgarian",
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "es": "spanish",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "hu": "hungarian",
    "it": "italian",
    "lt": "lithuanian",
    "lv": "latvian",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "sk": "slovak",
    "sl": "slovenian",
    "sv": "swedish",
}

# Enumerate the raw Europarl files in the data directory.
pairs      = []       # (english_path, lang_path, code)
found_lang = []
for code in language_codes:
    en_file  = data_dir / f"europarl-v7.{code}-en.en"
    xx_file  = data_dir / f"europarl-v7.{code}-en.{code}"
    if en_file.exists() and xx_file.exists():
        pairs.append((en_file, xx_file, code))
        found_lang.append(code)
    else:
        print(f"Missing files for {code}: skipped")
print(f"Found {len(found_lang)} languages: {', '.join(found_lang)}.")
print("The data is being processed, please wait.")

# Build a dictionary to store the raw data. 
# Key: en sentence 
# Value: dictionary of translations into other languages 
  # Key: language code xx
  # Value: translation into xx language
table = defaultdict(dict) # using dict allows to eliminate duplicates
for en_path, xx_path, code in pairs:
    with en_path.open(encoding="utf-8") as f_en, xx_path.open(encoding="utf-8") as f_xx:
        for en_line, xx_line in zip(f_en, f_xx):
            en = en_line.rstrip() # the english of xx language
            xx = xx_line.rstrip() # the translation into xx language
            table[en][code] = xx


# Only keep the sentences that appear in every single language
# and which have a character length greater than (or equal to) args.min_length.  
keep_en = [s for s, d in table.items() if len(d) == len(found_lang) and len(s) >= args.min_length] 

# Randomly shuffle the data.
r = random.Random(args.seed)
r.shuffle(keep_en)

# Write the parallel corpora to disk.
train_output = { "en": (out_dir / "train.en").open("w", encoding="utf-8", newline="\n") }
dev_output = { "en": (out_dir / "dev.en").open("w", encoding="utf-8", newline="\n") }
test_output = { "en": (out_dir / "test.en").open("w", encoding="utf-8", newline="\n") }
for code in found_lang:
    train_output[code] = (out_dir / f"train.{code}").open("w", encoding="utf-8", newline="\n")
    dev_output[code] = (out_dir / f"dev.{code}").open("w", encoding="utf-8", newline="\n")
    test_output[code] = (out_dir / f"test.{code}").open("w", encoding="utf-8", newline="\n")
for i, en in enumerate(keep_en):
    if i < args.num_dev:
        output = dev_output
    elif i < args.num_dev + args.num_test:
        output = test_output
    else:
        output = train_output
    output["en"].write(en + "\n") # write english file
    for code in found_lang:
        output[code].write(table[en][code] + "\n") # write each xx language file
for file in output.values():
    file.close()


