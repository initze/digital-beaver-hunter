from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely


def get_results(yolo_result):
    """
    get results from yolo prediction results output
    create a dataframe with all important information
    """
    res = yolo_result.cpu()
    names = res.names
    boxes = res.boxes
    classes = boxes.cls.numpy()
    confidence = boxes.conf.numpy()
    image_path = res.path
    image_name = Path(image_path).name
    image_id = Path(image_path).stem

    # Create Dataframe and fill
    df = pd.DataFrame()
    df["classes"] = np.array(classes, dtype=np.int8)
    df["confidence"] = confidence
    df["class_name"] = [names[c] for c in classes]
    df["image_id"] = image_id
    df["image_path"] = image_path
    df["image_name"] = image_name
    # get box coordinates
    df_boxes = pd.DataFrame(boxes.xywhn, columns=["x", "y", "width", "height"])

    return df.join(df_boxes)


def get_class_counts(df_results, image_list=None, get_max_proba=True):
    """
    Get class specific number of detected objects, from precalculated output dataframe
    """
    df_class_count = (
        df_results.groupby(by=["image_id", "class_name"])
        .count()
        .rename(columns={"index": "count"})[["count"]]
        .unstack()
    )
    df_maxconf = (
        df_results.groupby(by=["image_id", "class_name"])
        .max()
        .rename(columns={"confidence": "max_confidence"})[["max_confidence"]]
        .unstack()
    )
    df_class_count = df_class_count.join(df_maxconf)
    df_class_count.columns = df_class_count.columns.map("_".join).str.strip("|")
    if image_list:
        image_ids = [Path(im).stem for im in image_list]
        image_ids_add = pd.Series(image_ids)[
            ~pd.Series(image_ids).isin(df_class_count.index)
        ]
        df_image_ids_add = pd.DataFrame(index=image_ids_add)
        df_class_count = pd.concat([df_class_count, df_image_ids_add])

    return (
        df_class_count.replace(np.nan, 0)
        .sort_index()
        .reset_index(drop=False)
        .rename(columns={"index": "image_id"})
    )


def row_to_geom(row):
    with rasterio.open(row.raster_file) as src:
        rows, cols = src.shape
        ymin, ymax = row["y"] - (row["y2"] / 2), row["y"] + (row["y2"] / 2)
        xmin, xmax = row["x"] - (row["x2"] / 2), row["x"] + (row["x2"] / 2)
        uly, ulx = src.xy(ymax * rows, xmin * cols)
        lry, lrx = src.xy(ymin * rows, xmax * cols)
        # lrx, lry =  src.xy((row['y'] + row['y2']) * rows, (row['x'] + row['x2']) * cols)
    geom = shapely.geometry.box(uly, ulx, lry, lrx)
    return geom
