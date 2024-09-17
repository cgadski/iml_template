# IML Project Template

## Quickstart

1. [Install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). I recommend the Miniforge installer.
2. **Create a Conda "environment".** If you're using the CLI (recommended), `conda env create -n iml_env -f environment.yml` creates an environment with all the packages listed in `environment.yml`. 
    - You'll also need to activate the environment: `conda activate iml_env`.
    - (Maybe [read some docs on Conda](https://conda.io/projects/conda/en/latest/commands/env/create.html).)
4. **Launch Jupyterlab** (installed by Conda) by running `jupyter lab`. It should open in your web browser. 
    - See `example_assignment.ipynb` for an example notebook.
5. To export `$NOTEBOOK.ipynb` as a self-contained HTML document, run `make build/$NOTEBOOK.html`. **You can submit this**, along with any code it depends on, as your homework assignment.

## Tips

### Plaintext instead of notebooks?

`.ipynb` files (Jupyter notebooks) are JSON documents that you shouldn't edit manually. If you really like plaintext, you can use a `.py` file as a notebook with [the jupytext percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html). Jupytext can convert between `.py` files and `.pynb` files, and the `make build/$NOTEBOOK.html` target will still work.

### Reloading source files from notebooks?

If you're `import`ing a source file into a notebook, you will eventually notice that re-evaluating the `import` statement doesn't actually re-import the file, and so using a new version of your source file seems to require restarting the jupyter kernel.

You can get around this by asking Python to actually re-import the module using `importlib`. Fortunately there is also an easier way: add a cell
```
%load_ext autoreload
%autoreload 2
```
somewhere in your notebook to enable magical auto-reloading of your source files. See [here](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html).


