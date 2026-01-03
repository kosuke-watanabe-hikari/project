# src/setup.py
from setuptools import setup, find_packages

setup(
    name="mylib",          # パッケージ名
    version="0.1",
    packages=find_packages(),  # __init__.py があるフォルダを自動検出
)
