import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "Energy_Consumption_Prediction_MLModel"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data_loader.py",
    f"src/{project_name}/preprocessing.py",
    f"src/{project_name}/train.py",
    f"src/{project_name}/evaluate.py",
    f"src/{project_name}/Model.py",
    f"src/{project_name}/config/__init__.py",

    "Research/trials.ipynb",
    "templates/index.html"


]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")