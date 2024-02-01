import os
from CustomerChurn import logger
import pandas as pd
from CustomerChurn.utils.common import is_nan
from CustomerChurn.entity.config_entity import DataValidationConfig


class DataValidation:
    """
    A class for performing data validation based on provided configuration.
    """

    def __init__(self, config: DataValidationConfig) -> None:
        """
        Initializes the DataValidation instance with the provided configuration and loads data.

        Args:
        - config (DataValidationConfig): Configuration settings for data validation.
        """
        self.config = config
        self.data = pd.read_csv(self.config.unzip_data_dir)

    def _validate_num_columns(self) -> bool:
        """
        Validates the number of columns in the DataFrame.

        Returns:
        - bool: True if the number of columns matches the expected schema, False otherwise.
        """
        try:
            all_cols = list(self.data.columns)
            all_schema = list(self.config.all_schema.keys())

            validation_status = (all_cols == all_schema)

            return validation_status

        except Exception as e:
            raise e

    def _validate_type_columns(self) -> bool:
        """
        Validates the types of columns in the DataFrame.

        Returns:
        - bool: True if column types match the expected schema, False otherwise.
        """
        try:
            all_cols = list(self.data.columns)
            dtypes = self.data.dtypes
            all_schema = self.config.all_schema

            validation_status = True
            for i in all_cols:
                if dtypes[i] != all_schema[i]:
                    validation_status = False

            return validation_status

        except Exception as e:
            raise e

    def _validate_na_values(self) -> bool:
        """
        Validates the presence of NaN values in the DataFrame.

        Returns:
        - bool: True if NaN values are within the specified threshold, False otherwise.
        """
        try:
            total_rows = len(self.data)
            threshold = total_rows * self.config.nan_ratio
            all_cols = list(self.data.columns)
            na_values = self.data.isna().sum()

            validation_status = True
            for col in all_cols:
                if na_values[col] > threshold:
                    validation_status = False

            return validation_status

        except Exception as e:
            raise e

    def _validate_categories(self) -> bool:
        """
        Validates categorical columns in the DataFrame.

        Returns:
        - bool: True if all unique categories are present in the expected categories, False otherwise.
        """
        try:
            categories = self.config.categories
            data = self.data

            validation_status = True
            for col in categories:
                for category in data[col].unique():
                    if category not in categories[col] and not isnan(category):
                        validation_status = False

            return validation_status

        except Exception as e:
            raise e

    def final_validation(self) -> None:
        """
        Performs the final validation and writes the status to a file.

        Writes the validation status for number of columns, column types, NaN values,
        and categorical columns to a specified file.

        Returns:
            - None
        """
        try:
            validation_column = self._validate_num_columns()
            validation_types = self._validate_type_columns()
            validation_na = self._validate_na_values()
            validation_categories = self._validate_categories()

            if validation_column and validation_na and validation_categories and validation_types:
                validation_all = True
            else:
                validation_all = False

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation number of column status: {validation_column}\n"
                        f"Validation column types status: {validation_types}\n"
                        f"Validation NA values status: {validation_na}\n"
                        f"Validation categorical columns: {validation_categories}\n\n"
                        f"Validation all: {validation_all}")

        except Exception as e:
            raise e
