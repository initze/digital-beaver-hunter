import os
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from digital_beaver_hunter.utils.results import get_class_counts, get_results


def run_inference(
    data_dir: str,
    name: str,
    confidence: float,
    model: str,
    output_dir: str,
    imgsz: int = 2000,
):
    """
    Run YOLO inference on a set of images and save the results.

    This function performs object detection using a YOLO model on images in a specified directory,
    and saves the detection results and summary statistics.

    Parameters:
    data_dir (str): Directory containing the input images.
    name (str): Name of the project or dataset.
    confidence (float): Confidence threshold for object detection.
    model (str): Path to the YOLO model file.
    output_dir (str): Directory to save the output files.
    imgsz (int, optional): Input image size for the model. Defaults to 2000.

    Returns:
    None

    Outputs:
    - CSV and HTML files with detailed detection results for each image.
    - CSV and HTML files with summary statistics of detected objects per image.
    - Annotated images with detected objects.

    Notes:
    - Assumes input images are in JPG format.
    - Uses the YOLO model from the ultralytics package.
    - Saves both detailed results and summary statistics.
    - Creates output directories if they don't exist.
    """
    # setup images
    project_name = name
    image_dir = Path(data_dir) / project_name
    save_dir = Path(output_dir) / project_name
    save_dir_images = save_dir / "images"
    image_list = list(image_dir.glob("*.jpg"))

    model_path = model

    # setup model
    model = YOLO(model_path)

    # run prediction
    results = model(source=str(image_dir), conf=confidence, verbose=True, imgsz=imgsz)

    reslist = [get_results(res) for res in results]
    df_output = pd.concat(reslist).reset_index()

    df_class_count = get_class_counts(df_output, image_list=image_list)

    inference_images = df_output["image_path"].unique()

    model.predictor.save_dir = save_dir_images
    for image in inference_images[:]:
        results = model(source=image, conf=confidence, imgsz=imgsz, save=True)

    # check if reports dir exists
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Save outputs
    df_output.to_html(save_dir / "detected_features.html")
    df_output.to_csv(save_dir / "detected_features.csv", index=False)

    # save object count summary
    df_class_count.to_html(save_dir / "detected_image_summary.html")
    df_class_count.to_csv(save_dir / "detected_image_summary.csv", index=False)
