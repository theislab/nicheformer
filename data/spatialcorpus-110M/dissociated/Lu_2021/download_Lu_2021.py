import os 
import sys
from nicheformer.data.download import download_tar
from nicheformer.data.constants import GeneExpressionOmnibus, DefaultPaths

if len(sys.argv)==1:
    out_path = DefaultPaths.DISSOCIATED
else:
    out_path = sys.argv[1]

geo_id = 'GSE172127'

if not os.path.exists(f"{path}/raw"):
    os.mkdir(f"{path}/raw")

if not os.path.exists(f"{out_path}/raw/{geo_id}"):
    os.mkdir(f"{out_path}/raw/{geo_id}")

download_tar(
    f"{GeneExpressionOmnibus.DOWNLOAD_URL}{geo_id}{GeneExpressionOmnibus.FORMAT}", 
    save_path=f"{out_path}/raw/{geo_id}/raw_data_{geo_id}.tar", 
    file_format=["mtx", "tsv"]
)
