from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from digital_beaver_hunter.utils.geo import (
    get_global_coords_from_yolo_output,
    get_best_utm_epsg,
)
from digital_beaver_hunter.utils.geom import set_footprint


def process_stats_footprints(
    name: str, data_dir: str, base_dir_vectors: str, vector_suffix: str
):
    """
    Process statistics and footprints for a dataset, combining YOLO detection results with vector data.

    This function loads YOLO detection results, combines them with vector footprints,
    and produces various output files including vector data with detection statistics
    and individual feature locations.

    Parameters:
    name (str): Name of the dataset.
    data_dir (str): Directory containing the detection results.
    base_dir_vectors (str): Base directory for vector files.
    vector_suffix (str): Suffix for vector files.

    Returns:
    None

    Outputs:
    - A GeoPackage file with vector footprints and detection statistics.
    - A GeoPackage file with centroids of the above.
    - A GeoPackage file with individual feature locations in WGS84 (EPSG:4326).

    Notes:
    - Assumes detection results are stored in CSV files in the data_dir.
    - Vector files are expected to have either a 'Basename' or 'Name' column.
    - The function performs coordinate transformations and joins between vector and detection data.
    - Output files are saved in the same directory as the input detection results.
    """
    # setup paths
    ds_name = name
    save_dir = Path(data_dir) / ds_name

    # load inference results

    # class counts
    df_class_count = pd.read_csv(save_dir / "detected_image_summary.csv")
    # single features
    print((save_dir / "detected_features.csv").exists())
    df_features = pd.read_csv(save_dir / "detected_features.csv")

    # load vectors
    vector_dir = Path(base_dir_vectors)
    vector_file = vector_dir / ds_name / f"{ds_name}_{vector_suffix}"
    gdf = gpd.read_file(vector_file)

    # extract basename
    if "Basename" in gdf.columns:
        gdf["image_id"] = gdf["Basename"].str.replace(".macs", "")
    elif "Name" in gdf.columns:
        gdf["image_id"] = gdf["Name"].str.replace(".macs", "")
    # join (left)
    joined = gdf.set_index("image_id").join(df_class_count.set_index("image_id"))

    # setup output columns
    cols = list(df_class_count.columns.values)
    cols.append("geometry")

    gdf_out = joined.reset_index(drop=False)[cols].replace(np.nan, 0)

    # save files
    outfile = save_dir / (ds_name + "_vector.gpkg")
    gdf_out.to_file(outfile)

    # calculate centroids and save
    gdf_out_centroid = gdf_out.copy()
    gdf_out_centroid["geometry"] = gdf_out.centroid

    outfile_centroid = save_dir / (ds_name + "_vector_centroid.gpkg")
    print(outfile_centroid)
    gdf_out_centroid.to_file(outfile_centroid)

    # make local boxes

    # image ids with content
    image_ids = df_features["image_id"].unique()
    # find local utm zone
    epsg = get_best_utm_epsg(gdf)
    # filter to relevant footprints
    gdf_filtered_projected = gdf[gdf["image_id"].isin(image_ids)].to_crs(epsg)
    gdf_filtered_projected["geometry"] = gdf_filtered_projected.apply(
        lambda row: set_footprint(row), axis=1
    )
    # convert local to global coords
    global_geoms = []
    error_features = []
    for i, row in df_features.iterrows():
        try:
            global_geoms.append(
                get_global_coords_from_yolo_output(row, gdf_filtered_projected)
            )
        except:
            print(f"Error in row {i}")
            error_features.append(i)
            global_geoms.append(None)
    print(f"Features {error_features} could not be resolved")

    # create output feature gdf
    gdf_features = gpd.GeoDataFrame(
        df_features, geometry=global_geoms, crs=f"EPSG:{epsg}"
    ).to_crs(4326)

    features_outfile = save_dir / (ds_name + "_feature_locations.gpkg")
    gdf_features.to_file(features_outfile)

    return True
