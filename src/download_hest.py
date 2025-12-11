import os
import zipfile
import datasets
from huggingface_hub import snapshot_download
from tqdm import tqdm

def download_hest(patterns, local_dir):
    repo_id = 'MahmoodLab/hest'
    snapshot_download(repo_id=repo_id, allow_patterns=patterns, repo_type="dataset", local_dir=local_dir)

    seg_dir = os.path.join(local_dir, 'cellvit_seg')
    if os.path.exists(seg_dir):
        print('Unzipping cell vit segmentation...')
        for filename in tqdm([s for s in os.listdir(seg_dir) if s.endswith('.zip')]):
            path_zip = os.path.join(seg_dir, filename)
                        
            with zipfile.ZipFile(path_zip, 'r') as zip_ref:
                zip_ref.extractall(seg_dir)

if __name__ == "__main__":
    local_dir = 'hest_data' # hest will be dowloaded to this folder
    ids_to_query = ['TENX96', 'TENX99'] # list of ids to query

    list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    print(f"Downloading patterns: {list_patterns}")
    download_hest(list_patterns, local_dir)
