import argparse
import os
import time

import pandas as pd
from tqdm import tqdm


def convert_to_megabytes(size):
    if pd.notna(size):
        size = size.lower()  # Convert to lowercase to handle various cases
        if "kb" in size:
            return float(size.replace("kb", "")) / 1024  # 1 Kb = 1/1024 Mb
        elif "mb" in size:
            return float(size.replace("mb", ""))
        elif "gb" in size:
            return float(size.replace("gb", "")) * 1024
    return 0.0  # Return 0 for NaN or unsupported formats


def extract_data(geo_path, save_dir):
    """
    Extract data from GEO excel file
    """
    # pandas read a xlsx file
    df_geos = pd.read_excel(geo_path, sheet_name="GEO samples")
    # set geo_accession as index
    df_geos = df_geos.set_index("geo_accession")

    # iterate over geo_accession ids
    for geo_accession in tqdm(list(df_geos.index.unique())):
        try:
            df = pd.read_html(
                f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_accession}", match="Supplementary", header=0
            )[-1]
            max_size = df["Size"].apply(convert_to_megabytes).max()
            df_geos.loc[geo_accession, "max_file_size"] = max_size

            formats = list(df["File type/resource"].dropna().unique())
            df_geos.loc[geo_accession, "formats"] = ", ".join(formats)
        except:
            print("Error: {geo_accession}")
        time.sleep(0.1)

    df_geos = df_geos.sort_values(["max_file_size", "geo_accession"], ascending=[False, True])

    df_geos.to_csv(os.path.join(save_dir, "GEO_samples_scsimilarity_extracted.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GEO data from an Excel file.")
    parser.add_argument("--geo_excel_path", help="Path to the input Excel file")
    parser.add_argument("--save_dir", help="Path to save output file")
    args = parser.parse_args()

    extract_data(args.geo_excel_path, args.save_dir)
