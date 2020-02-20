===========
Work Sample
===========

This projects loads the data samples and the POI data from the Data folder and solve the four questions in the question list. The output files are stored in the Output folder.
The python version is 3.6.9 and for Docker was 3.6.10

***************
Installation
***************
.. code:: python3

   python3 setup.py install

***************
Usage example
***************
This example can be used as follows:

.. code:: python3

   python3 work_sample.py

This script loads two csv files and generate output png files and csv file in the Output folder

*****************
Development setup
*****************
To install all development dependencies:

.. code:: python3

   pip3 install -r requirements.txt

*************
Docker run
*************
Run these commands for Docker:
.. code::

   docker build --no-cache -t work_sample:v1 .

   docker run -it --rm work_sample:v1 bash

   docker run -it --rm -v ${PWD}/new_output:/usr/src/work/Output work_sample:v1

*****************
Release History
*****************

    - 0.0.0
        - The first release

*****************
Meta
*****************
Hilda Faraji  – hilda015@gmail.com
