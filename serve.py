# Default Flask app :)

import os
import random
import urllib.request

import quickdraw
from flask import Flask, render_template, request

app = Flask(__name__)


def gen_random_name_png(ln):
    return ''.join([str(random.randint(0, 9)) for x in range(ln)]) + ".png"


@app.route('/')
def index():
    return render_template("index.html")

@app.route("/class", methods=["POST"])
def classif():
    data = request.data.decode()
    resp = urllib.request.urlopen(data)
    fname = gen_random_name_png(15)
    with open(fname, "wb") as f:
        f.write(resp.file.read())
        f.close()
    res = quickdraw.classif(fname)
    os.remove(fname)
    return res

app.run(debug=True)