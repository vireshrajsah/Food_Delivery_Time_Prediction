import os
from setuptools import setup,find_packages

REQUIREMENTS_FILE_NAME = 'requirements.txt'
HYPHEN_E_DOT = '-e .'

def getrequirements(requirements_file_path):
    with open(requirements_file_path, 'r') as file:
        requirements_list = file.readlines()
        requirements_list = [i.replace('\n', '') for i in requirements_list]
        requirements_list = [i for i in requirements_list if i!='']
        if HYPHEN_E_DOT in requirements_list:
            requirements_list.remove(HYPHEN_E_DOT)
        return requirements_list

setup(
    author ="Viresh Raj Sah",
    name ="Food Delivery Time Predictor",
    version ="0.0.1",
    author_email ="viresh.raj.sah@gmail.com",
    install_requires = getrequirements(requirements_file_path= os.path.join(os.getcwd(),REQUIREMENTS_FILE_NAME)),
    packages= find_packages()
)
