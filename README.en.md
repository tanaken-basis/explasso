# Explasso

[English](README.en.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [日本語](README.jp.md)

## Overview
This is a program in Python to generate an experimental design using a sparse modeling technique.

## Usage
- Without a Python runtime environment
    - On Windows, download "[explasso_gradio_win.7z](https://github.com/tanaken-basis/explasso/raw/master/explasso_gradio_win.7z)" and decompress it, double-click "explasso_gradio.exe" to start the web application for automatically generating experimental designs. It may take 2-3 minutes to start the web application. This web application runs on a local machine.
        - If "Windows protected your PC", click "More Info" and click "Run Anyway".
    - On Mac (Apple silicon), download "[explasso_gradio_mac.7z](https://github.com/tanaken-basis/explasso/raw/master/explasso_gradio_mac.7z)" and decompress it, double-click "explasso_gradio" to start the web application for automatically generating experimental designs. It may take 2-3 minutes to start the web application. This web application runs on a local machine.
        - If you cannot run the application on your Mac, type `sudo spctl --master-disable` in the terminal to allow all applications to run. You can return to the original state by typing `sudo spctl --master-enable`.
- Using Python programs
    - Running "explasso.py" in Python generates your experimental design automatically. It saves the results as a CSV file. 
    - By running "explasso_gradio.py" in Python, you can start the web application created by Gradio to generate experimental designs automatically. You can adjust parameters for automated generation, execute calculations to generate results and save calculated results to files. You can run the web application on a local machine.
    - The same program is also written in the notebook file "explasso.ipynb". 

## Notes on using Python programs
- This code uses the Python library CVXOPT to compute optimization. Please refer to https://github.com/cvxopt/cvxopt for installation information.
- This code uses the Python library Gradio to create the web app. Please refer to https://github.com/gradio-app/gradio for installation information.

## Sample of the execution screen

![alt text](explasso_gradio-1.jpg)
