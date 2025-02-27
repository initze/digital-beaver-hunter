import os
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

from digital_beaver_hunter.utils.inference import run_inference
from digital_beaver_hunter.utils.postprocessing import process_stats_footprints

BASEDIR = Path("/isipd/projects-noreplica/p_initze/yolov8_object_detection")
MODEL = BASEDIR / "models/v15i.yolov5pytorch_yolov10x_scratch_finetune.pt"
N_JOBS = 4
CUDA_ENV = "CUDA_VISIBLE_DEVICES='6'"
BASEDIR_VECTORS = "/isipd/projects/Response/Restricted_Airborne/MACS/Canada/2023_Perma-X_Canada/1_MACS_original_images"
VECTOR_SUFFIX = "footprints_full.shp"
output_dir = Path(".") / "output/v15i.yolov5pytorch_yolov10x_scratch_finetune"
CONFIDENCE = 0.1


def run_project(
    data_dir, project_name, model, output_dir, confidence=CONFIDENCE, env=""
):
    if not Path(output_dir).exists():
        os.makedirs(output_dir)
    # run inference
    run_inference(
        data_dir=data_dir,
        name=project_name,
        model=model,
        output_dir=output_dir,
        confidence=confidence,
    )
    # run stats and documentation
    process_stats_footprints(
        name=project_name,
        data_dir=output_dir,
        base_dir_vectors=BASEDIR_VECTORS,
        vector_suffix=VECTOR_SUFFIX,
    )


data_dir = BASEDIR / "data"
dirlist = list(data_dir.glob("*"))
projects = [d.name for d in dirlist if d.is_dir()]

projects_run = [
    p for p in projects[:] if not (Path(output_dir) / p).exists() and " " not in p
]
# print(projects_run)

# run hardcoded only test area
projects_run = ["20230711-233224_056_UWPingos_01_1000m"]
Parallel(n_jobs=N_JOBS)(
    delayed(run_project)(data_dir, project, MODEL, output_dir, env=CUDA_ENV)
    for project in tqdm(projects_run[:])
)
