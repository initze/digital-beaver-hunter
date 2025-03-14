import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

import typer
from joblib import Parallel, delayed
from tqdm import tqdm
from typer_config.decorators import use_yaml_config
from typing_extensions import Annotated

from digital_beaver_hunter.utils.inference import run_inference
from digital_beaver_hunter.utils.postprocessing import process_stats_footprints

app = typer.Typer()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_project(
    data_dir: Path,
    project_name: str,
    model: Path,
    output_dir: Path,
    confidence: float,
    device: int,
    basedir_vectors: str,
    vector_suffix: str,
    image_size: int,
    classes: Optional[List[int]],  # New parameter
    logger: Optional[logging.Logger] = None,  # New parameter
    delete_input: bool = False,  # New parameter
) -> bool:
    try:
        if not output_dir.exists():
            os.makedirs(output_dir)
        # Run inference
        if logger:
            logger.info(f"Running inference for project: {project_name}")
        inference_complete = run_inference(
            data_dir=data_dir,
            name=project_name,
            model=model,
            output_dir=output_dir,
            confidence=confidence,
            device=device,
            imgsz=image_size,
            classes=classes,  # Add classes parameter
            logger=logger,
        )

        # check if first step was correctly finished
        if not inference_complete:
            if logger:
                logger.info(f"Processing of inference failed!: {project_name}")
            return False
        # Run stats and documentation
        if logger:
            logger.info(
                f"Processing stats and documentation for project: {project_name}"
            )
        footprints_complete = process_stats_footprints(
            name=project_name,
            data_dir=output_dir,
            base_dir_vectors=basedir_vectors,
            vector_suffix=vector_suffix,
        )
        if logger:
            logger.info(f"Completed processing for project: {project_name}")
    except Exception as e:
        if logger:
            logger.error(f"Error processing {project_name}: {e}")
        return False

    # delete input
    if delete_input:
        logger.info(f"WARNING: Deleting input data for project {project_name} is activated!")
        if all([inference_complete, footprints_complete]):
            if logger:
                logger.info("Processing successful.")
                logger.info(f"Deleting input data for project: {project_name}")
            try:
                # logger.info("Deletion not activated yet!")
                shutil.rmtree(data_dir / project_name)
                if logger:
                    logger.info(
                        f"Successfully deleted input data for project: {project_name}"
                    )
            except Exception as e:
                if logger:
                    logger.error(f"Error deleting input data for {project_name}: {e}")
        else:
            logger.info("Deleting input data is activated, but processing failed! \nInput data are NOT deleted! ")
    else:
        logger.info("Deleting input data is deactivated!")
    
    return True


@app.command()
@use_yaml_config()
def main(
    basedir_data: Path = typer.Option(..., help="Base directory for the project"),
    model: Path = typer.Option(..., help="Path to the model file"),
    n_jobs: int = typer.Option(8, help="Number of parallel jobs"),
    device: int = typer.Option(6, help="Device number for inference"),
    basedir_vectors: str = typer.Option(..., help="Base directory for vector files"),
    vector_suffix: str = typer.Option(
        "footprints_full.shp", help="Suffix for vector files"
    ),
    output_dir: Path = typer.Option(Path(".") / "output", help="Output directory"),
    confidence: float = typer.Option(0.1, help="Confidence threshold"),
    image_size: int = typer.Option(2016, help="Image size for inference"),
    projects_to_run: Optional[List[str]] = typer.Option(
        None, help="List of projects to run (optional)"
    ),
    n_datasets: Optional[int] = typer.Option(
        None, help="Number of datasets to process (optional)"
    ),
    filter_startswith: Optional[str] = typer.Option(
        None, help="Filter projects that start with this string (optional)"
    ),
    classes: Optional[List[int]] = typer.Option(
        None, help="List of class IDs to detect (optional)"
    ),  # New parameter
    logfile: Path = typer.Option(
        Path("logs/app.log"), help="Path to the log file (optional)"
    ),
    delete_input: Annotated[bool, typer.Option("--delete-input")] = False,
):
    data_dir = basedir_data
    dirlist = list(data_dir.glob("*"))
    projects = [d.name for d in dirlist if d.is_dir()]

    if projects_to_run:
        projects_run = projects_to_run
    else:
        # Check if dir exists
        projects_run = [
            p for p in projects if not (output_dir / model.stem / p).exists()
        ]
        # Then, apply the filter_startswith if it's not None
        if filter_startswith is not None:
            projects_run = [p for p in projects_run if p.startswith(filter_startswith)]

    # Filter to fixed number of projects
    if n_datasets is not None:
        projects_run = projects_run[:n_datasets]

    if logger:
        logger.info(f"Projects to run: {projects_run}")

    # Add file handler to logger
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    Parallel(n_jobs=n_jobs)(
        delayed(run_project)(
            data_dir,
            project,
            model,
            output_dir / model.stem,
            confidence,
            device,
            basedir_vectors,
            vector_suffix,
            image_size,
            classes,
            logger=logger,
            delete_input=delete_input,    # Add classes parameter
        )
        for project in tqdm(projects_run)
    )

    # Remove file handler from logger
    logger.removeHandler(file_handler)


if __name__ == "__main__":
    app()
