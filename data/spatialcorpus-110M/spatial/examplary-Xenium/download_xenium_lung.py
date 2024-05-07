import os 
import sys
from nicheformer.data.constants import DefaultPaths
from nicheformer.data.download import download_zip


if len(sys.argv)==1:
    out_path = DefaultPaths.SPATIAL
else:
    out_path = sys.argv[1]

if not os.path.exists(f"{out_path}/raw"):
    os.mkdir(f"{out_path}/raw")

download_zip(
    url="https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.zip",
    save_path=f"{out_path}/raw",
    fn="Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs"
)