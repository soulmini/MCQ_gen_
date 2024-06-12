from setuptools import find_packages , setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    package=find_packages(),
    install_requires = ["openai", "langchain", "streamlit", "python-dotenv", "PyPDF2", "python"]
)