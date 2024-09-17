# IML Project Template

## Quickstart

1. [**Install Conda.**](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) I recommend the Miniforge installer.
2. **Create a Conda "environment".** If you're using the CLI (recommended), `conda env create -n iml_env -f environment.yml` creates an environment with all the packages listed in `environment.yml`. 
    - You'll also need to activate the environment: `conda activate iml_env`.
    - (Maybe [read some docs on Conda](https://conda.io/projects/conda/en/latest/commands/env/create.html).)
3. **Launch Jupyterlab** (installed by Conda) by running `jupyter lab`. It should open in your web browser. 
    - See `example_assignment.ipynb` for an example notebook.
4. **Export your notebook to HTML** with `make build/$NOTEBOOK.html`.

## Tips

### Plaintext instead of notebooks?

`.ipynb` files (Jupyter notebooks) are JSON documents that you shouldn't edit manually. If you really like plaintext, you can use Python source file as a notebook with [the jupytext percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html). Jupytext can convert between `.py` files and `.pynb` files, and the `make build/$NOTEBOOK.html` target will still work. Some editors will recognize this format and let you run code blocks in a Jupyter kernel. (I've tried VSCode and Zed personally, and surely this is possible in emacs and vim.) See `example_plaintext.py` for an example plaintext notebook.

### Reloading source files from notebooks?

If you're `import`ing a source file into a notebook, you will eventually notice that re-evaluating the `import` statement doesn't actually re-import the file, and so using a new version of your source file seems to require restarting the jupyter kernel. Meh!

You can ask Python to actually re-import a module using [`importlib.reload`](https://docs.python.org/3/library/importlib.html#importlib.reload). Fortunately there is also an easier way: add a cell
```
%load_ext autoreload
%autoreload 2
```
somewhere in your notebook to make Jupyter magically auto-reload your source files. See [the docs on autoreload](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html).


