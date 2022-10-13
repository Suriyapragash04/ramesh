import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("home.html")

@app.route("/predict", methods = ["POST"])
def predict():

        perf_6_month_avg = float(request.form["perf_6_month_avg"])
        perf_12_month_avg = float(request.form["perf_12_month_avg"])
        binary_pieces_past_due = float(request.form["binary_pieces_past_due"])
        binary_local_bo_qty = float(request.form["binary_local_bo_qty"])
        national_inv = float(request.form["national_inv"])
        lead_time = float(request.form["lead_time"])
        in_transit_qty = float(request.form["in_transit_qty"])
        forecast_3_month = float(request.form["forecast_3_month"])
        local_bo_qty = float(request.form["local_bo_qty"])


        forecast_6_month = float(request.form["forecast_6_month"])
        forecast_9_month = float(request.form["forecast_9_month"])
        potential_issue = float(request.form["potential_issue"])
        pieces_past_due = float(request.form["pieces_past_due"])
        min_bank = float(request.form["min_bank"])
        sales_1_month = float(request.form["sales_1_month"])
        sales_3_month = float(request.form["sales_3_month"])
        sales_6_month = float(request.form["sales_6_month"])



        sales_9_month = float(request.form["sales_9_month"])
        deck_risk = float(request.form["deck_risk"])
        oe_constraint = float(request.form["oe_constraint"])
        ppap_risk = float(request.form["ppap_risk"])
        stop_auto_buy = float(request.form["stop_auto_buy"])
        rev_stop = float(request.form["rev_stop"])


  
        feature=[national_inv,lead_time,in_transit_qty,forecast_3_month,
        forecast_6_month,forecast_9_month,sales_1_month,sales_3_month,sales_6_month,
        sales_9_month,min_bank,potential_issue,pieces_past_due,perf_6_month_avg,
        perf_12_month_avg,local_bo_qty,deck_risk,oe_constraint,ppap_risk,
        stop_auto_buy,rev_stop,binary_pieces_past_due,binary_local_bo_qty]
        features1 = [np.array(feature)]
        prediction = model.predict(features1)

        if prediction == 0:
            output = 'Not went on backorder'
        elif prediction == 1:
            output = 'Went on backorder'

        return render_template('home.html', prediction_text="{}".format(output))







if __name__ == "__main__":
    app.run(debug=True)
'''# model.predict([[perf_6_month_avg,perf_12_month_avg,binary_pieces_past_due,
        binary_local_bo_qty,national_inv,lead_time,in_transit_qty,forecast_3_month,
        forecast_6_month,forecast_9_month,potential_issue,pieces_past_due,min_bank,
        sales_1_month,sales_3_month,sales_6_month,sales_9_month,deck_risk,
        oe_constraint,ppap_risk,stop_auto_buy,rev_stop]])'''