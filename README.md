## Clone the repository

```
git clone https://github.com/rodolphe-reynard/291Dproject.git
```

## Build the environment

```
uv sync
```

**If you also want to run .ipynb files:**

```
uv sync --all-extras
```

## Modules

### Create modules

Create a python file in the `src/project291d/xxx` folder. The name of the file will be the name of the module.

### Import modules

If you create a file called `my_module.py`, you can import it in your code as follows:

```python
from project291d.xxx import my_module
```

Then you can use the functions, classes and variables defined in `my_module.py` in your code as follows:

```python
f = my_module.my_function()
```

### Import objects from modules

You can also import specific objects from a module as follows:

```python
from project291d.xxx.my_module import my_function
```

Then you can use the function `my_function` in your code as follows:

```python
f = my_function()
```

## Architecture

![Architecture](resources/architecture.png)
