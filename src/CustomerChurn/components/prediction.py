from CustomerChurn import logger
from CustomerChurn.utils.common import load_bin, save_json, save_txt, get_size
import pandas as pd
import numpy as np
import urllib.request as request
from CustomerChurn.entity.config_entity import PredictionConfig
import os
from pathlib import Path


class Prediction:
    """
    A class for making predictions using a trained model and vectorizer.
    """

    def __init__(self, config: PredictionConfig):
        """
        Initializes the Prediction instance with the provided configuration.

        Args:
        - config (PredictionConfig): Configuration settings for making predictions.
        """
        self.config = config

    def _download_file(self, url, datapath):
        """
        Downloads a file from a given URL and saves it to a specified path if not already present.

        Args:
        - url (str): URL of the file to download.
        - datapath (str): Path where the downloaded file should be saved.

        Returns:
        - None
        """
        if not os.path.exists(datapath):
            filename, headers = request.urlretrieve(url=url, filename=datapath)
            logger.info(f"{filename} download!")
            save_txt(
                data=str(headers),
                path=Path(os.path.join(self.config.root_dir, "download_status.txt")),
            )
        else:
            logger.info(f"File already exists of size: {get_size(Path(datapath))}")

    def predict(self):
        """
        Uses the trained model and vectorizer to make predictions on new data.

        Returns:
        - None
        """
        model = load_bin(Path(self.config.model_path))
        vectorizer = load_bin(Path(self.config.vectorizer_path))
        try:
            data = pd.read_csv(self.config.data_path)
        except Exception as e:
            logger.info(f"error - {e} while access predict data")
            self._download_file(
                "https://raw.githubusercontent.com/izam-mohammed/data-source/main/sample_customer_churn.csv",
                os.path.join(self.config.root_dir, "sample_data.csv"),
            )
            data = pd.read_csv(os.path.join(self.config.root_dir, "sample_data.csv"))

        data[self.config.target_column] = np.array(["Yes"])

        vectorized_data = vectorizer.transform(data)
        vectorized_data = vectorized_data[:, :-1]
        prediction = model.predict(vectorized_data)
        logger.info(f"predicted the new data as {prediction[0]}")

        save_json(
            path=self.config.prediction_file, data={"prediction": float(prediction[0])}
        )
