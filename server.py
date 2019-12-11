import argparse
from flask import Flask, request, send_from_directory, render_template
import curate_data as cda
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser()
args = parser.parse_args()

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

if __name__ == "__main__":
	app.run()
