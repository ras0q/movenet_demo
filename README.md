# movenet_demo

## Requirement

- Windows 11
- (Recommended) [rye](https://github.com/mitsuhiko/rye)
- (Recommended) VSCode

## Run demo

```sh
$ python .\src\main.py --help
usage: main.py [-h] [--device DEVICE] [--width WIDTH] [--height HEIGHT]

options:
  -h, --help       show this help message and exit
  --device DEVICE  camera device index
  --width WIDTH    camera width
  --height HEIGHT  camera height
```

### With rye

1. Install rye referring to <https://rye-up.com/guide/installation>

2. Install Python with rye

```sh
rye toolchain fetch python
```

3. Install dependencies

```sh
rye sync
```

4. Run demo

```sh
rye run python src/main.py
```

### Without rye

> **Warning**
> You can just run this demo without rye, but it's hard to develop.

1. Install Python according to `.python-version`
2. (Recommended) Setup virtual environment (e.g. venv, pipenv, etc.)
3. Install dependencies

```sh
pip install .
```

4. Run demo

```sh
python src/main.py
```
