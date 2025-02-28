import os
from pathlib import Path
from typing import List, Optional

import typer
from joblib import Parallel, delayed
from tqdm import tqdm
from typer_config.decorators import use_yaml_config

from digital_beaver_hunter.utils.inference import run_inference
from digital_beaver_hunter.utils.postprocessing import process_stats_footprints

app = typer.Typer()


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
) -> bool:
    try:
        if not output_dir.exists():
            os.makedirs(output_dir)
        # run inference
        run_inference(
            data_dir=data_dir,
            name=project_name,
            model=model,
            output_dir=output_dir,
            confidence=confidence,
            device=device,
            imgsz=image_size,
        )
        # run stats and documentation
        process_stats_footprints(
            name=project_name,
            data_dir=output_dir,
            base_dir_vectors=basedir_vectors,
            vector_suffix=vector_suffix,
        )
    except Exception as e:
        print(f"Error processing {project_name}: {e}")
        return False
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
):
    data_dir = basedir_data
    dirlist = list(data_dir.glob("*"))
    projects = [d.name for d in dirlist if d.is_dir()]

    if projects_to_run:
        projects_run = projects_to_run
    else:
        projects_run = [
            p
            for p in projects
            if not (output_dir / p).exists() and p.startswith("2023")
        ]

    if n_datasets is not None:
        projects_run = projects_run[:n_datasets]

    print("Projects to run:", projects_run)

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
        )
        for project in tqdm(projects_run)
    )


if __name__ == "__main__":
    app()
