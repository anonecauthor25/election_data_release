import re
import numpy as np
from os import path
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm.contrib.concurrent import thread_map

class DataMerger:
    """
    A class for merging JSON files from multiple subfolders into a consolidated Pandas DataFrame.

    The DataMerger class processes JSON files, extracts relevant information,
    and combines the data into a single DataFrame. This class allows the user
    to configure the file selection based on folder path, number of dates,
    date range, model filtering, and other criteria. Additionally, it supports
    multithreading for efficient processing of large datasets.

    Class Attributes:
    -----------------
    _big_df : pd.DataFrame or None
        The final merged DataFrame after processing. Initially set to None.
    _folder_path : str or None
        The directory path containing subfolders with JSON files. Set via `initialize`.
    _max_workers : int or None
        The number of threads to use for parallel processing. Set via `initialize`.
    _start_date : str or None
        The start date for filtering files. Set via `initialize`.
    _end_date : str or None
        The end date for filtering files. Set via `initialize`.
    _models : list or None
        The list of models to filter by. Set via `initialize`.
    _filter_by : str or None
        Filtering mode for selecting dates. Can be "random", "last", "range", or "all". Set via `initialize`.

    Methods:
    --------
    initialize(cls, folder_path, max_workers=8, start_date=None, end_date=None, models=None, filter_by="range"):
        Initializes the class by setting the folder path, maximum workers, date range, model list,
        and filter criteria.

    get_df(cls, override=False):
        Returns the merged DataFrame. If the DataFrame has not been created or `override`
        is True, it triggers the merging process.

    _merge_data(cls):
        Private method that handles the merging of JSON files from the specified folder path.
        It filters files by date and applies multithreaded processing to build the final DataFrame.
    """

    _big_df = None
    _folder_path = None
    _max_workers = None
    _start_date = None
    _end_date = None
    _nb_dates = None
    _models = None
    _filter_by = None

    @classmethod
    def initialize(cls, folder_path, max_workers=8, nb_dates=10, start_date=None, end_date=None, models=None, filter_by="last"):
        """
        Initializes the DataMerger class with user-defined parameters.

        Parameters:
        -----------
        folder_path : str
            Path to the directory containing subfolders of JSON files.
        max_workers : int, optional
            Maximum number of worker threads to use for processing (default is 8).
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format for filtering (default is None).
        end_date : str, optional
            End date in 'YYYY-MM-DD' format for filtering (default is None).
        models : list of str, optional
            List of models to filter (default is None, meaning all models).
        filter_by : str, optional
            Filter criteria for selecting dates. Can be "range", "random", "last", or "all" (default is "range").
        """
        cls._folder_path = folder_path
        cls._max_workers = max_workers
        cls._filter_by = "range" if start_date or end_date is not None else filter_by
        cls._start_date = pd.to_datetime(start_date) if start_date else None
        cls._end_date = pd.to_datetime(end_date) if end_date else None
        cls._models = models if models else None
        cls._nb_dates = nb_dates

    @classmethod
    def get_df(cls, override=False):
        """
        Returns the merged DataFrame, creating it if it doesn't exist or if `override` is True.

        Parameters:
        -----------
        override : bool, optional
            If True, forces re-merging of data even if the DataFrame has already been created (default is False).

        Returns:
        --------
        pd.DataFrame
            The merged DataFrame containing data from all specified JSON files.

        Raises:
        -------
        ValueError
            If the folder path has not been set by calling `initialize()` first.
        """
        if cls._big_df is None or override:
            if cls._folder_path is None:
                raise ValueError("Folder path not set. Call initialize() first.")
            cls._big_df = cls._merge_data()
        return cls._big_df

    @classmethod
    def _merge_data(cls):
        """
        Private method that merges data from JSON files within the specified folder.

        The method filters the files based on the selected date range and model filtering,
        and processes them in parallel using multithreading. Each JSON file is parsed into
        a DataFrame and tagged with its corresponding date and model identifier.

        Returns:
        --------
        pd.DataFrame
            A consolidated DataFrame with data from all relevant JSON files.

        Raises:
        -------
        AssertionError
            If `filter_by` is not one of ["all", "random", "last", "range"].
        """
        all_file_paths = [x for x in glob(path.join(cls._folder_path, "*/*/*.json")) if "serp-google" not in x]
        all_dates = list(set([path.split("/")[-1].strip(".json") for path in all_file_paths]))

        assert cls._filter_by in ["all", "random", "last", "range"], "Choose from ['all', 'random', 'last', 'range'] for filter_by argument"

        if cls._filter_by == "random":
            dates = np.random.choice(all_dates, cls._nb_dates, replace=False)
        elif cls._filter_by == "last":  # take from last
            dates = sorted(all_dates, key=lambda p: pd.to_datetime(p), reverse=True)[:cls._nb_dates]
        elif cls._filter_by == "range":  # filter by date range
            if cls._start_date and cls._end_date:
                dates = [d for d in all_dates if cls._start_date <= pd.to_datetime(d) <= cls._end_date]
            else:
                raise ValueError("Both start_date and end_date must be set for 'range' filter.")
        else:  # No filter, get all dates
            dates = all_dates

        # Filter models if specified
        if cls._models:
            file_paths = [p for p in all_file_paths if any(p.split("/")[-2] == model for model in cls._models) and any(date in p for date in dates)]
        else:
            file_paths = [p for p in all_file_paths if any(date in p for date in dates)]

        def process_file(file_path):
            """
            Processes a single JSON file and returns its data as a DataFrame.

            Parameters:
            -----------
            file_path : str
                Path to the JSON file to be processed.

            Returns:
            --------
            pd.DataFrame
                A DataFrame containing the data from the JSON file, with added `date`
                and `model` columns.
            """
            model = file_path.split("/")[-2]
            str_date = re.search(r"(\d{4}-\d{2}-\d{2}).json", str(file_path)).group(1)
            curr_df = pd.read_json(file_path)
            curr_df["date"] = pd.to_datetime(str_date)
            curr_df["model"] = model
            return curr_df

        # Parallel processing of response files
        data_frames = thread_map(process_file, file_paths, max_workers=cls._max_workers)
        data_frames = [df for df in data_frames if df is not None]
        big_df = pd.concat(data_frames, ignore_index=True)

        # Reorder columns and clean data
        big_df = big_df[["model", "date"] + [col for col in big_df.columns if col not in ["model", "date"]]]
        big_df.sort_values(["model", "date"], inplace=True)
        big_df.reset_index(drop=True, inplace=True)
        big_df['response'] = big_df['response'].replace(r'^\s*$', pd.NA, regex=True)  # replace empty strings or only whitespaces by Na

        # Map unique questions and responses to IDs
        id_to_question = dict(enumerate(big_df['question'].unique()))
        question_to_id = {v: k for k, v in id_to_question.items()}
        big_df["question_id"] = big_df["question"].map(question_to_id)

        id_to_response = dict(enumerate(big_df["response"].unique()))
        response_to_id = {v: k for k, v in id_to_response.items()}
        big_df["response_id"] = big_df["response"].map(response_to_id)

        return big_df


