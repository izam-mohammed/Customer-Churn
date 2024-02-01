import os
import urllib.request as request
import zipfile
from pathlib import Path
from CustomerChurn import logger
from CustomerChurn.utils.common import get_size, save_txt
from CustomerChurn.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """
    A class for handling data ingestion tasks based on provided configuration.
    """

    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initializes the DataIngestion instance with the provided configuration.

        Args:
        - config (DataIngestionConfig): Configuration settings for data ingestion.
        """
        self.config = config

    def download_file(self) -> None:
        """
        Downloads the data file from the specified source URL.

        If the file already exists, logs the file size without downloading it.

        Returns:
            None
        """
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL, filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded!")
            save_txt(
                data=str(headers),
                path=Path(os.path.join(self.config.root_dir, "download_status.txt")),
            )
        else:
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
            )

    def extract_zip_file(self) -> None:
        """
        Extracts the contents of a zip file into the specified directory.

        If the directory doesn't exist, it will be created.

        Returns:
        None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
