# Flower Extension featuring Global Update Optimizer

This introductory example uses the simulation capabilities of Flower to simulate a large number of clients on either a single machine of a cluster of machines.

## Running the example (via Jupyter Notebook)

Coming soon

## Running the example

Start by cloning the code example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/guo-tensorflow . && rm -rf flower && cd guo-tensorflow
```

This will create a new directory called `guo-tensorflow` containing the following files:

```
-- README.md       <- Your're reading this right now
-- sim.py          <- Example code
-- pyproject.toml  <- Example dependencies
```

### Installing Dependencies

Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import tensorflow"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

### Run Federated Learning Example

```bash
poetry run python3 sim.py
```
