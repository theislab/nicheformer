import requests
import os
import tarfile
import h5py
from tqdm import tqdm
import argparse
import zipfile

def format_files(members, file_format):
    for tarinfo in members:
        if os.path.splitext(tarinfo.name.removesuffix(".gz"))[1] == "." + file_format:
            yield tarinfo

def download_zip(
    url,
    save_path, 
    fn, 
):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for any errors in the request

    zip_fn = f"{save_path}/{fn}.zip"

    # Open a local file for writing in binary mode.
    with open(zip_fn, "wb") as file:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            if chunk:  # Filter out keep-alive new chunks.
                file.write(chunk)

    print(f"Downloaded, saved to {zip_fn}")
    with zipfile.ZipFile(zip_fn,"r") as zip_ref:
        zip_ref.extractall(f"{save_path}/{fn}")

def download_tar(
    url,
    save_path="/lustre/groups/ml01/projects/2023_nicheformer/dissociated_database_raw",
    file_format=None
):
    """
    Download an h5ad file from a given URL, add 'donor_id' and 'tissue' columns to adata.obs, and save it to the specified location.

    Args:
        url (str): The URL of the h5ad file to download.
        save_path (str): The local path where the downloaded file will be saved.
        donor_id_value (str): The value to set for the 'donor_id' column.
        tissue_value (str): The value to set for the 'tissue' column.
        counts (str): name of the layer where counts are stored, otherwise None
        organism (str): 'human' or 'mouse'
    """
    try:
        
        # Send an HTTP GET request to the URL to retrieve the file content.
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for any errors in the request

        # Open a local file for writing in binary mode.
        with open(save_path, "wb") as file:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                if chunk:  # Filter out keep-alive new chunks.
                    file.write(chunk)

        print(f"Downloaded {url}, saved to {save_path}")
        
        tar = tarfile.open(save_path)
        if file_format is None:
            tar.extractall(path=os.path.dirname(save_path))
        else:
            print(type(file_format))
            if isinstance(file_format, list):
                for form in file_format:
                    print(form)
                    tar.extractall(path=os.path.dirname(save_path), members=format_files(tar, form))
            else:
                tar.extractall(path=os.path.dirname(save_path), member=format_files(tar, file_format))
        tar.close()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset by GEO ID')
    parser.add_argument('--geo-id', type=str)
    parser.add_argument('--save-path', type=str, 
                        default="/lustre/groups/ml01/projects/2023_nicheformer/dissociated_database_raw")
    args = parser.parse_args()
    url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=" + args.geo_id + "&format=file"
    download_tar(url, args.save_path)