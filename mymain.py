from flask import Flask, render_template, request, flash, jsonify,redirect,url_for
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
from wtforms import Form,TextField,StringField,FloatField
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.models import load_model


app = Flask(__name__)
app.config['SECRET_KEY'] = "Donottellanyone"
UPLOAD_FOLDER = '/home/mrunalj/PycharmProjects/ml/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class annform(Form):
    ambtemp = FloatField('ambtemp')
    modtemp = FloatField('modtemp')
    solirr = FloatField('solirr')


@app.route("/")
def index():

    return render_template('ss.html')


@app.route("/ann")
def ann():
    return render_template("ann.html")


@app.route('/ann_form', methods=['GET', 'POST'])
def ann_form():
    if request.method== 'POST':

        list = []
        ambtemp = request.form['ambtemp']
        modtemp = request.form['modtemp']
        solirr = request.form['solirr']


        # sc = StandardScaler()
        import annload as obj
        sc = obj.load()
        myarray = [ambtemp, modtemp, solirr]
        print(ambtemp,modtemp,solirr)
        myarray = np.array([myarray]).astype('float32')

        myarray = sc.transform(myarray)
        model = load_model('annmodel.h5')

        ans = model.predict(myarray)

        myans = str(ans[0][0])

        print(myans)
    return render_template('ann.html', value=myans)


@app.route("/lstm")
def lstm():
    return render_template("lstm.html")


@app.route('/lstm_hour',methods=['GET', 'POST'])
def lstm_hour():
    if request.method == 'POST':


    #filename=request.form['browse']
        f=request.files['browse']
        filename=secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if 'hour' in request.form:
            import lstmload as ld

            model = load_model('hour_ahead_regressor.h5')

            dataset1 = pd.read_csv("/home/mrunalj/PycharmProjects/ml/uploads/"+filename)

            dataset_power_zero1 = dataset1[dataset1['Date'] == '0']
            dataset1 = dataset1.drop(dataset_power_zero1.index, axis=0)
            data = dataset1.iloc[:, 2:6].values
            data = data.astype('float32')
            length = len(data)

            transformed_set = ld.sc.transform(data)

            newx = []
            y = []

            for i in range(110, length):
                newx.append(transformed_set[i - 110:i, :])
                y.append(data[i, -1])

            newx, y = np.array(newx), np.array(y)

            newx = np.reshape(newx, (newx.shape[0], newx.shape[1], 4))

            pred = model.predict(newx)
            pred = ld.sc_predict.inverse_transform(pred)

            plt.clf()
            plt.plot(pred, color='blue', label="predicted power")
            plt.plot(y, color='red', label="actual power")

            figure = plt.gcf()
            figure.savefig('static/plot_hour.png', dpi=100)

            return render_template("lstm_graph.html", ans=pred[-1])

        if 'day' in request.form:
            import lstmload as ld

            model1 = load_model('model.h5py')

            dataset1 = pd.read_csv("/home/mrunalj/PycharmProjects/ml/uploads/" + filename)

            dataset_power_zero1 = dataset1[dataset1['Date'] == '0']
            dataset1 = dataset1.drop(dataset_power_zero1.index, axis=0)
            data = dataset1.iloc[:, 2:6].values
            data = data.astype('float32')
            length = len(data)

            transformed_set = ld.sc.transform(data)

            newx = []
            y = []
            for i in range(320, length - 16):
                newx.append(transformed_set[i - 320:i, :])
            newx = np.array(newx)
            newx = np.reshape(newx, (newx.shape[0], newx.shape[1], 4))
            pred = model1.predict(newx)
            pred = ld.sc_predict.inverse_transform(pred)

            plt.clf()
            plt.plot(pred[-1], color='blue', label="predicted power")
            avg=sum(pred[-1]) / float(len(pred[-1]))
            # plt.plot(y , color = 'red' , label = "actual power")

            figure1 = plt.gcf()
            figure1.savefig('static/plot1_day.png', dpi=100)

            return render_template("lstm_day.html", value=avg)

        if 'week' in request.form:
            import lstmload as ld

            model1 = load_model('modelweek.h5py')

            dataset1 = pd.read_csv("/home/mrunalj/PycharmProjects/ml/uploads/" + filename)

            dataset_power_zero1 = dataset1[dataset1['Date'] == '0']
            dataset1 = dataset1.drop(dataset_power_zero1.index, axis=0)
            data = dataset1.iloc[:, 2:6].values
            data = data.astype('float32')
            length = len(data)

            transformed_set = ld.sc.transform(data)

            newx = []
            y = []
            for i in range(224, length - 112):
                newx.append(transformed_set[i - 224:i, :])
            newx = np.array(newx)
            newx = np.reshape(newx, (newx.shape[0], newx.shape[1], 4))
            pred = model1.predict(newx)
            pred = ld.sc_predict.inverse_transform(pred)
            plt.clf()

            avg = sum(pred[-1]) / float(len(pred[-1]))
            plt.plot(pred[-1], color='blue', label="predicted power")
    # plt.plot(y , color = 'red' , label = "actual power")

            figure2 = plt.gcf()
            figure2.savefig('static/plot1_week.png', dpi=100)

        return render_template("lstm_week.html", value=avg)


if __name__ == "__main__":
    app.run()