# Explasso

[English](README.en.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [日本語](README.jp.md)

## Overview
This is a program in Python to generate an experimental design by using a sparse modeling technique.

## Usage
- Running "explasso.py" in Python generates your experimental design automatically. It saves the results as a CSV file. There is also a program written in a notebook called "explasso.ipynb".
- By running "otg_gradio.py" in Python, you can start the web application created by Gradio for automatically generating experimental designs. You can adjust parameters for automated generation, execute calculations to generate results, and save calculated results to files. This web application can be run on a local machine.
- The same program is also written in the notebook file "explasso.ipynb". 

## Notes on using Python programs
- In this code, we use the Python library CVXOPT to compute optimization. Please refer to https://github.com/cvxopt/cvxopt for installation information.
- In this code, we use the Python library Gradio to create the web app. Please refer to https://github.com/gradio-app/gradio for installation information.

## Sample of execution screen

![alt text](explasso_gradio-1.jpg)
