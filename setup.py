from setuptools import setup

setup(
    name='Work Sample',
    version='0.1.0',
    py_modules=["Work_Sample"],
    license='NO',
    description='Statistical Analysis of Work-Sample-Data',
    long_description=open('README.rst').read(),
    install_requires=['numpy','pandas','geopandas','sklearn','tabulate','matplotlib','scipy','statsmodels','descartes'],
    url='https://github.com/hilda',
    author='Hilda Faraji',
    author_email='hilda015@gmail.com'
)
