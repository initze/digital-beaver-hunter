#!/usr/bin/env python
# coding: utf-8

import typer

from digital_beaver_hunter.utils.inference import run_inference

app = typer.Typer()


@app.command()
def main(
    data_dir: str = typer.Option("data", help="Directory containing input data"),
    name: str = typer.Option(..., help="Name of the project"),
    conf: float = typer.Option(0.14, help="Confidence threshold for detection"),
    model: str = typer.Option(
        "models/yolov8l_MACS_beaver_v10best.pt", help="Path to the model file"
    ),
    output_dir: str = typer.Option("output", help="Directory for output data"),
):
    run_inference(data_dir, name, conf, model, output_dir)


if __name__ == "__main__":
    app()
