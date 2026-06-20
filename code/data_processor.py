import pandas as pd
import os
from typing import List, Dict, Any, Tuple

class DataProcessor:
    def __init__(self, input_csv: str, output_csv: str):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Loads the input CSV file into a Pandas DataFrame."""
        if not os.path.exists(self.input_csv):
            raise FileNotFoundError(f"Input file not found: {self.input_csv}")
        self.df = pd.read_csv(self.input_csv)
        return self.df

    def get_image_paths(self, row: pd.Series) -> List[str]:
        """Extracts and cleans image paths from a DataFrame row."""
        raw_paths = row.get("image_paths", "")
        if pd.isna(raw_paths) or not isinstance(raw_paths, str):
            return []
        
        # Split by semicolon and remove empty strings
        paths = [p.strip() for p in raw_paths.split(";") if p.strip()]
        return paths

    def append_result(self, row_index: int, result_dict: Dict[str, Any]):
        """Appends the resulting fields back to the DataFrame."""
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call load_data() first.")
        
        for key, value in result_dict.items():
            self.df.at[row_index, key] = value

    def save_output(self):
        """Saves the DataFrame to the output CSV."""
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Cannot save.")
        
        # Ensure the specified columns are saved in the output
        columns_to_save = [
            "user_id", "image_paths", "user_claim", "claim_object",
            "evidence_standard_met", "evidence_standard_met_reason", 
            "risk_flags", "issue_type", "object_part", 
            "claim_status", "claim_status_justification", 
            "supporting_image_ids", "valid_image", "severity"
        ]
        
        # Add missing columns if they don't exist
        for col in columns_to_save:
            if col not in self.df.columns:
                self.df[col] = None

        # Reorder to match desired output
        out_df = self.df[columns_to_save]
        out_df.to_csv(self.output_csv, index=False)
        print(f"Data saved to {self.output_csv}")
