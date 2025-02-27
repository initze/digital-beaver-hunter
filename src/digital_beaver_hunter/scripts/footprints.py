
import typer

from digital_beaver_hunter.utils.postprocessing import process_stats_footprints

app = typer.Typer()


@app.command()
def main(
    name: str = typer.Option(..., help="Name of the dataset"),
    data_dir: str = typer.Option("output", help="Directory for output data"),
    base_dir_vectors: str = typer.Option(
        "/isipd/projects/Response/Restricted_Airborne/MACS/2021_Perma-X_Alaska/01_raw_data",
        help="Base directory for vector files",
    ),
    vector_suffix: str = typer.Option(
        "footprints_full.shp", help="Suffix for vector files"
    ),
):
    process_stats_footprints(name, data_dir, base_dir_vectors, vector_suffix)


if __name__ == "__main__":
    app()
