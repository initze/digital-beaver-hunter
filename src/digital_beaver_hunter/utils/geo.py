import numpy as np
from shapely.geometry import Polygon


def yolo_to_projected_polygon(image_coords, yolo_coords):
    """
    Convert YOLO format bounding box coordinates to a projected polygon.

    This function takes the coordinates of an image's corners in a projected
    coordinate system and a bounding box in YOLO format, and returns a Shapely
    Polygon representing the bounding box in the projected coordinate system.

    Parameters:
    image_coords (list of tuples): List of 4 tuples representing the coordinates
                                   of the image corners in the order:
                                   (lower-left, upper-left, upper-right, lower-right)
    yolo_coords (tuple): Tuple containing the YOLO format bounding box coordinates
                         in the order: (x_center, y_center, width, height)
                         These should be normalized values between 0 and 1.

    Returns:
    shapely.geometry.Polygon: A Polygon representing the bounding box in the
                              projected coordinate system.

    Note:
    - The input image_coords should be in a projected coordinate system.
    - The YOLO coordinates are assumed to be normalized (0 to 1).
    - The function assumes the image is not rotated in the projected space.
    """
    # Extract image coordinates
    ll, ul, ur, lr = image_coords[
        :4
    ]  # lower-left, upper-left, upper-right, lower-right

    # Create vectors for width and height of the image in projected space
    width_vector = np.array(ur) - np.array(ul)
    height_vector = np.array(ll) - np.array(ul)

    # Extract YOLO coordinates (normalized)
    x_center, y_center, width, height = yolo_coords

    # Calculate the corners of the YOLO box in normalized coordinates
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    # Calculate the projected coordinates of the YOLO box
    lower_left = np.array(ul) + x_min * width_vector + y_max * height_vector
    upper_left = np.array(ul) + x_min * width_vector + y_min * height_vector
    upper_right = np.array(ul) + x_max * width_vector + y_min * height_vector
    lower_right = np.array(ul) + x_max * width_vector + y_max * height_vector

    # Create a polygon from these coordinates
    polygon = Polygon(
        [tuple(lower_left), tuple(upper_left), tuple(upper_right), tuple(lower_right)]
    )

    return polygon


def get_global_coords_from_yolo_output(row, gdf_images):
    """
    Convert YOLO output coordinates to global projected coordinates.

    This function takes a row of YOLO detection results and a GeoDataFrame of image footprints,
    and returns a polygon in the global projected coordinate system representing the detected object.

    Parameters:
    row (pandas.Series or dict-like): A row containing YOLO detection results.
                                      Must include 'x', 'y', 'height', 'width', and 'image_id' fields.
                                      'x', 'y', 'height', and 'width' should be normalized YOLO coordinates.
    gdf_images (geopandas.GeoDataFrame): A GeoDataFrame containing image footprints.
                                         Must have 'image_id' as index or column and a 'geometry' column
                                         with Polygon or MultiPolygon geometries representing image footprints.

    Returns:
    shapely.geometry.Polygon: A Polygon representing the detected object in the global
                              projected coordinate system of the input image footprints.

    Notes:
    - The input GeoDataFrame (gdf_images) should be in a projected coordinate system.
    - The function assumes that the YOLO coordinates are normalized (0 to 1).
    - The function uses the yolo_to_projected_polygon function to perform the coordinate transformation.
    """

    yolo_coords = row[["x", "y", "height", "width"]].values
    row_image = gdf_images.set_index("image_id").loc[row["image_id"]]
    image_coords = list(dict.fromkeys(row_image.geometry.exterior.coords))[:4]
    geom_out = yolo_to_projected_polygon(image_coords, yolo_coords)
    return geom_out
