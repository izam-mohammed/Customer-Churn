import os
from pathlib import Path
import logging
import urllib.request as request

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")


project_name = "CustomerChurn"


list_of_files = [
    ".github/workflows/.gitkeep",
    ".github/FUNDING.yml",
    ".github/workflows/greetings.yml",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    f"research/{project_name}.ipynb",
    "research/trials.ipynb",
    "saved_models/__init__.py",
    "tests/test.py",
    "problem_statement.docx",
    "templates/index.html",
    "templates/404.html",
    "SECURITY.md",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "README.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    ".gitignore",
    "LICENSE",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists")

# writing data
def get_data(data_url, data_name):
    os.makedirs("data", exist_ok=True)
    request.urlcleanup()
    file_name, headers = request.urlretrieve(data_url, data_name)

project_head = "Customer Churn Prediction"
repo_name = "Customer-Churn"
short_description = "A repository for predict the customer churn data"

get_data("https://raw.githubusercontent.com/izam-mohammed/data-source/main/setup_env.py", "write.py")
os.system(f"python write.py {project_head} __ {repo_name} __ {short_description}")
os.remove("write.py")
