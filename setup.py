from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

    setup(
        name='Energy Consumopotion Prediction',
        version='1.0.0', 
        description='This is a Machine learning model for predicting energy consumption',
        author='Anirudh Johare',
        author_email='anirudhjohare@gmail.com')