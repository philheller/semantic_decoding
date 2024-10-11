# Setup

Copy the `fever_factual_final.jsonl` file from the [FactualityPrompt repo](https://github.com/nayeon7lee/FactualityPrompt.git) to this directory.
Follow the installation steps below in order to be able to use the wiki database.

Other files I had to change:
1. in the [`README.md`](https://github.com/nayeon7lee/FactualityPrompt/blob/b9424e838e17c4943ed1161b7993b9220d8ff593/README.md) of `fever_athene`, the command to build the database is wrong (only works if the fever_athene is built). So there, I just changed it:
    ```diff
    - PYTHONPATH=fever_athene python3 fever_athene/scripts/build_db_kilt.py data/knowledgesource.json data/kilt_db.db
    + PYTHONPATH=fever_athene python3 fever_athene/src/scripts/build_db_kilt.py data/knowledgesource.json data/kilt_db.db
    ```

2. I also had the same issue for importing common in the `fever_athene/scripts/build_db_kilt.py` file. I had to change the import to:
    ```diff
    - from common.util import load_jsonl
    + from src.common.util import load_jsonl
    ```

3. ⚠️ The [`README.md`](https://github.com/nayeon7lee/FactualityPrompt/blob/b9424e838e17c4943ed1161b7993b9220d8ff593/README.md) of `FactualityPrompt` gives instructions to download to this file (via wget): `kilt_knowledgesource.json`. Be sure to adapt the following command which is with `knowledgesource.json`instead of `kilt_knowledgesource.json`:
    ```diff
    - PYTHONPATH=fever_athene python3 fever_athene/scripts/build_db_kilt.py data/knowledgesource.json data/kilt_db.db
    - PYTHONPATH=fever_athene python3 fever_athene/scripts/build_db_kilt.py data/kilt_knowledgesource.json data/kilt_db.db
    ```

## Installation
The setup is not trivial for users on Windows, since the FactualityPrompt and also the Fever repos are not kept up to date, nor have they been built to handle Windows (Linux and MacOS may be fine).

Here are some things I encountered:
1. The packages that need to be installed are not properly marked by the FactualityPrompt repo. I created a separate env for just this experiment, as the packages are not really kept up to date.

2. The packages fever-* have issues with Windows encoding. For some packages I had to download them, unpack them and build them with altered code:

```bash
pip download fever-drqa
tar -xvf fever-drqa-0.1.0.tar.gz

# change the __init__.py to use utf-8 encoding for all with open() calls

python setup.py install
```
Here are the diffs for the python files which did not work on Windows:
```diff
- with open(file_path, 'r') as f:
+ with open(file_path, 'r', encoding='utf-8') as f:
```

3. A few packages only use PosixPath which may not work on Windows. I had to change the code to use pathlib.Path instead. For example in the `fever-drqa` package, I had to change the following code in the `__init__.py` file:
```diff
- from pathlib import PosixPath
+ from pathlib import Path

# ...


DATA_DIR = (
    os.getenv('DRQA_DATA') or
-    os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'data')
+    os.path.join(Path(__file__).absolute().parents[1], 'data')
)
```

The only parts needed for the generation in line with the experiment is the `get_wiki_from_db` function, which was copied over for simplicity. To build the database and for the usage of the FactualityPrompt, the Fever repo is used and a separate environment is created for that (compartmentalization).

## Env file

Since the entire setup is not maintained anymore and many things do not work out of the box, I decided to add the `env.yml` file which is my env exported. I have not tested this extensively, different OSs may behave differently. But this may help to get started.