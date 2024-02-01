from flask import Flask, request, jsonify
from CustomerChurn.config.configuration import ConfigurationManager
from CustomerChurn.pipeline.prediction import PredictionPipeline
from CustomerChurn.utils.common import load_json
import pandas as pd
from pathlib import Path
from CustomerChurn import logger
import os

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    return "Use API call to /predict_data"


@app.route("/train", methods=["GET", "POST"])
def training():
    os.system("python main.py")
    return "Training Successful!"


# API call
@app.route("/predict_data", methods=["POST", "GET"])
def pred():
    try:
        gender = request.args.get("gender")
        SeniorCitizen = request.args.get("SeniorCitizen")
        Partner = request.args.get("Partner")
        Dependents = request.args.get("Dependents")
        tenure = request.args.get("tenure")
        PhoneService = request.args.get("PhoneService")
        MultipleLines = request.args.get("MultipleLines")
        InternetService = request.args.get("InternetService")
        OnlineSecurity = request.args.get("OnlineSecurity")
        OnlineBackup = request.args.get("OnlineBackup")
        DeviceProtection = request.args.get("DeviceProtection")
        TechSupport = request.args.get("TechSupport")
        StreamingTV = request.args.get("StreamingTV")
        StreamingMovies = request.args.get("StreamingMovies")
        Contract = request.args.get("Contract")
        PaperlessBilling = request.args.get("PaperlessBilling")
        PaymentMethod = request.args.get("PaymentMethod")
        MonthlyCharges = request.args.get("MonthlyCharges")
        TotalCharges = request.args.get("TotalCharges")

        logger.info("\n\nX" + "==" * 16 + "Predicting" + "==" * 16 + "X")

        config = ConfigurationManager()
        config = config.get_prediction_config()
        data_path = config.data_path

        data = pd.DataFrame(
            {
                "Gender": [gender],
                "SeniorCitizen": [SeniorCitizen],
                "Partner": [Partner],
                "Dependents": [Dependents],
                "tenure": [tenure],
                "PhoneService": [PhoneService],
                "MultipleLines": [MultipleLines],
                "InternetService": [InternetService],
                "OnlineSecurity": [OnlineSecurity],
                "OnlineBackup": [OnlineBackup],
                "DeviceProtection": [DeviceProtection],
                "TechSupport": [TechSupport],
                "StreamingTV": [StreamingTV],
                "StreamingMovies": [StreamingMovies],
                "Contract": [Contract],
                "PaperlessBilling": [PaperlessBilling],
                "PaymentMethod": [PaymentMethod],
                "MonthlyCharges": [MonthlyCharges],
                "TotalCharges": [TotalCharges],
            }
        )
        data.to_csv(data_path, index=False)

        obj = PredictionPipeline()
        obj.main()

        file = load_json(path=Path(config.prediction_file))
        predict = file["prediction"]
        logger.info("\nX" + "==" * 38 + "X\n")

        return jsonify({"result": bool(predict)})

    except Exception as e:
        print("The Exception message is: ", e)
        return jsonify({"result": "error"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
