# Explasso

[English](README.en.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [日本語](README.jp.md)

## 概要
実験計画を自動生成するためのPythonのプログラムです。スパースモデリングの手法を用いています。

## 使い方
- "explasso.py"をPythonで実行すると、実験計画を自動生成します。自動生成結果をCSVファイルとして保存します。
- "explasso_gradio.py"をPythonで実行すると、GradioによるWebアプリが起動します。実験計画の自動生成のためのパラメータの調整、自動生成の計算の実行、計算結果のファイルへの出力などができます。このWebアプリは、ローカルで動作させることができます。
- ノートブックのファイル"explasso.ipynb"にも、同様の動作をするプログラムを記述しています。

## Pythonプログラムを使用する場合の注意点
- このコードでは、最適化の計算のために、Pythonライブラリ CVXOPT を使用しています。CVXOPT のインストールについては、https://github.com/cvxopt/cvxopt を参照してください。
- Webアプリの作成には、Pythonライブラリ Gradio を使用しています。Gradio のインストールについては、https://github.com/gradio-app/gradio を参照してください。

## サンプルの実行結果

![alt text](explasso_gradio-1.jpg)
