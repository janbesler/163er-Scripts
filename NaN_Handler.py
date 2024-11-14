import pandas as pd
from datetime import timedelta
from collections import defaultdict

class NaNHandler:
    def __init__(self, df, time_column='datetime', interpolate_threshold=timedelta(minutes=1)):
        """
        Initialize NaNHandler with a DataFrame and optional parameters for the time column and interpolation threshold.
        
        Parameters:
        - df (pd.DataFrame): The input DataFrame with potential NaN values.
        - time_column (str): The name of the time column (default: 'datetime').
        - interpolate_threshold (timedelta): Maximum duration for gaps to be interpolated (default is 1 minute).
        """
        self.df = df.copy()
        self.time_column = time_column
        self._set_time_index()
        self.threshold = interpolate_threshold

    def _set_time_index(self):
        """
        Automatically detects and sets the time column as the DataFrame index.
        Converts string datetime columns to datetime format if necessary.
        Determines the period of the datetime index.
        """
        # Check if the current index is datetime-like
        if pd.api.types.is_datetime64_any_dtype(self.df.index):
            print("Time column already set as index.")
        else:
            # Check if specified time_column exists in columns
            if self.time_column in self.df.columns:
                if not pd.api.types.is_datetime64_any_dtype(self.df[self.time_column]):
                    self.df[self.time_column] = pd.to_datetime(self.df[self.time_column], errors='coerce')
                self.df.set_index(self.time_column, inplace=True)
                print(f"Time column '{self.time_column}' set as index.")
            else:
                # Look for any datetime-like column if time_column is not found
                for col in self.df.columns:
                    if pd.api.types.is_datetime64_any_dtype(self.df[col]) or \
                       (self.df[col].apply(lambda x: isinstance(x, str)).all() and \
                        pd.to_datetime(self.df[col], errors='coerce').notna().all()):
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        self.df.set_index(col, inplace=True)
                        print(f"Detected and set '{col}' as the datetime index.")
                        break
                else:
                    raise ValueError("No suitable time column found to set as index.")

        # Detect the frequency of the datetime index
        self._detect_period()

    def _detect_period(self):
        """
        Detects the period (frequency) of the datetime index. If multiple unique periods
        are found, it prints them and selects the period that occurs most frequently.
        """
        # Calculate the difference between consecutive timestamps
        time_diffs = self.df.index.to_series().diff().dropna()

        # Count occurrences of each unique time difference (period)
        period_counts = time_diffs.value_counts()

        if len(period_counts) > 1:
            print("Multiple unique period values detected:")
            for period, count in period_counts.items():
                print(f"Period: {period}, Count: {count}")

            # Select the most frequent period
            most_common_period = period_counts.idxmax()
            print(f"Using the most common period: {most_common_period}")
        else:
            most_common_period = period_counts.index[0]
            print(f"Detected period: {most_common_period}")

        self.period = most_common_period

    def analyze_nan_gaps(self):
        """
        Analyzes gaps of consecutive NaN values in each column, identifying gaps shorter and longer than the threshold.
        
        Returns:
        - results (dict): Summary of NaN gap analysis.
        """
        results = {}
        
        # Get indices of NaN values for each column
        nan_indices_per_column = {col: self.df[col][self.df[col].isna()].index.tolist() for col in self.df.columns if self.df[col].isna().sum() > 0}

        for col, timestamps in nan_indices_per_column.items():
            short_gaps_count = 0
            long_gaps = defaultdict(int)
            current_gap = [timestamps[0]]  # Initialize the first gap

            # Identify gaps by calculating differences between consecutive timestamps
            for i in range(1, len(timestamps)):
                # Check if the current timestamp is consecutive based on the detected period
                if timestamps[i] - timestamps[i - 1] == self.period:
                    current_gap.append(timestamps[i])
                else:
                    # End of a gap
                    gap_duration = current_gap[-1] - current_gap[0] + self.period
                    
                    if gap_duration <= self.threshold:
                        short_gaps_count += 1
                    else:
                        long_gaps[gap_duration] += 1

                    # Start a new gap
                    current_gap = [timestamps[i]]
            
            # Check the last gap after the loop
            if current_gap:
                gap_duration = current_gap[-1] - current_gap[0] + self.period
                if gap_duration <= self.threshold:
                    short_gaps_count += 1
                else:
                    long_gaps[gap_duration] += 1

            # Sort long gaps by length and extract the three longest gaps
            sorted_long_gaps = sorted(long_gaps.items(), key=lambda x: x[0], reverse=True)
            top_three_long_gaps = sorted_long_gaps[:3]
            total_long_gaps_count = sum(long_gaps.values())

            # Store results
            results[col] = {
                "short_gaps_count": short_gaps_count,
                "total_long_gaps_count": total_long_gaps_count,
                "top_three_long_gaps": top_three_long_gaps
            }
            
            # Print summary for long gaps
            print(f"\nColumn: {col}")
            print(f"Number of gaps shorter than {self.threshold}: {short_gaps_count}")
            print(f"Total number of gaps longer than {self.threshold}: {total_long_gaps_count}")
            print("Top three longest gaps and their occurrences:")
            for gap_length, count in top_three_long_gaps:
                print(f"{gap_length}: {count} times")

    def interpolate_small_gaps(self):
        """
        Interpolates NaN values for gaps up to a specified duration (self.threshold).
        
        Returns:
        - df_interpolated (pd.DataFrame): The DataFrame with interpolated NaN values for small gaps.
        """
        df_interpolated = self.df.copy()
        
        for col in df_interpolated.columns:
            is_nan = df_interpolated[col].isna()
            gap_lengths = is_nan.groupby((~is_nan).cumsum()).apply(lambda x: len(x) if x.all() else 0)
            
            # Interpolate only small gaps
            for start_idx, gap_length in zip(gap_lengths.index, gap_lengths):
                if 0 < gap_length <= self.threshold / self.period:
                    # Interpolate for the identified gap positions
                    gap_positions = is_nan[is_nan.groupby((~is_nan).cumsum()).transform('size') == gap_length].index
                    df_interpolated[col].loc[gap_positions] = df_interpolated[col].interpolate(method='time').loc[gap_positions]
        
        return df_interpolated
