# Setup
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