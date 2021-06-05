from flask import Flask,request,render_template,redirect,url_for
import pickle
import numpy as np

model = pickle.load(open('iris.pkl','rb'))
app = Flask(__name__)

@app.route("/")
def man():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def home():
    slength = request.form["sLength"]
    swidth  = request.form["sWidth"]
    plength = request.form["pLength"]
    pwidth  = request.form["pWidth"]
    data = np.array([[slength,swidth,plength,pwidth]])
    pre = model.predict(data)
    return render_template('after.html',data=pre[0])

if __name__ == "__main__":
    app.run(debug=True)