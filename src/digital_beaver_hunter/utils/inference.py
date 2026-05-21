import os
from pathlib import Path
import logging
from typing import Optional
import time

import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

from digital_beaver_hunter.utils.results import get_class_counts, get_results

# Create a logger instance at the module level
logger = logging.getLogger(__name__)


def run_inference(
    data_dir: str,
    name: str,
    confidence: float,
    model: str,
    output_dir: str,
    imgsz: int = 2000,
    device: int = 0,
    classes: list = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
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
    device (int, optional): cuda device for inference
    classes (list, optional): List of class IDs to detect.
    logger (Optional[logging.Logger], optional): Logger to use for logging.

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
    start_time = time.time()
    if logger:
        logger.info(f"Starting inference for project: {name}")

    # setup images
    project_name = name
    image_dir = Path(data_dir) / project_name
    save_dir = Path(output_dir) / project_name
    save_dir_images = save_dir / "images"
    image_list = list(image_dir.glob("*.jpg"))

    if len(image_list) == 0:
        if logger:
            logger.info(f"No images found in directory: {image_dir}")
        return False

    if logger:
        logger.info(f"Total images to process: {len(image_list)}")

    model_path = model

    # setup model
    if logger:
        logger.info(f"Loading model: {model_path.stem}")
    model = YOLO(model_path)

    # run prediction
    reslist = []
    for image in tqdm(image_list):
        result = model(
            source=image,
            conf=confidence,
            verbose=False,
            imgsz=imgsz,
            device=device,
            classes=classes,
        )[0]
        reslist.append(pd.DataFrame(get_results(result)))

    df_output = pd.concat(reslist).reset_index()

    df_class_count = get_class_counts(df_output, image_list=image_list)

    inference_images = df_output["image_path"].unique()

    model.predictor.save_dir = save_dir_images

    processed_images = 0
    for image in tqdm(inference_images[:]):
        try:
            results = model(
                source=image,
                conf=confidence,
                imgsz=imgsz,
                save=True,
                device=device,
                verbose=False,
                classes=classes,
            )
            processed_images += 1
        except Exception as e:
            if logger:
                logger.error(f"Error processing image {image}: {e}")
            else:
                print(f"Error processing image {image}: {e}")
            continue

    if logger:
        logger.info(f"Total images with detected objects: {processed_images}")

    # check if reports dir exists
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Save outputs
    df_output.to_html(save_dir / "detected_features.html")
    df_output.to_csv(save_dir / "detected_features.csv", index=False)

    # save object count summary
    df_class_count.to_html(save_dir / "detected_image_summary.html")
    df_class_count.to_csv(save_dir / "detected_image_summary.csv", index=False)

    end_time = time.time()
    if logger:
        logger.info(f"Inference completed in {end_time - start_time:.2f} seconds")
    return True
