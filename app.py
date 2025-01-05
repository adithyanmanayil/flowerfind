from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np 

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def man():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
	data1 = request.form['a'];
	data2 = request.form['b'];
	data3 = request.form['c'];
	data4 = request.form['d'];
	arr = np.array([[data1, data2, data3, data4]])
	pred = model.predict(arr);
	if pred == 0:
			species = "Iris Setosa"
	elif pred == 1:
			species = "Iris Versicolor"
	else:
			species = "Iris Virginica"

	return render_template('home.html', data = species)

if __name__ == "__main__":
	app.run(debug = True)