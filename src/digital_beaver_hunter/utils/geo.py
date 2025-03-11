import numpy as np
from shapely.geometry import Polygon

import numpy as np
from shapely.geometry import Polygon
from math import radians, sin, cos

def order_rectangle_corners(polygon, yaw):
    """
    Order the corners of a rectangle polygon based on flight direction (yaw).
    
    Parameters:
    polygon (shapely.geometry.Polygon): Rectangle polygon in EPSG:4326 with 5 vertices
    yaw (float): Yaw angle in degrees, representing flight direction
    
    Returns:
    shapely.geometry.Polygon: New polygon with ordered corners 
    [upper left, upper right, lower right, lower left]
    """
    # Extract coordinates, removing the last (duplicate) point
    coords = np.array(list(polygon.exterior.coords)[:-1])
    
    # Handle 3D coordinates by taking only the first two dimensions (lat/lon)
    if coords.shape[1] == 3:
        coords_2d = coords[:, :2]
        has_z = True
    else:
        coords_2d = coords
        has_z = False
    
    # Calculate centroid
    centroid = np.mean(coords_2d, axis=0)
    
    # Convert yaw to radians
    yaw_rad = radians(yaw)
    
    # Create rotation matrix for -90 degrees from yaw (to get "up" direction)
    rotation_matrix = np.array([
        [cos(yaw_rad - np.pi/2), -sin(yaw_rad - np.pi/2)],
        [sin(yaw_rad - np.pi/2), cos(yaw_rad - np.pi/2)]
    ])
    
    # Function to rotate point around centroid
    def rotate_point(point):
        vector = point - centroid
        rotated = np.dot(rotation_matrix, vector)
        return rotated[1]  # Return y-coordinate after rotation
    
    # Sort points based on rotated y-coordinate (highest first)
    sorted_coords = sorted(coords_2d, key=rotate_point, reverse=True)
    
    # Separate upper and lower points
    upper = sorted_coords[:2]
    lower = sorted_coords[2:]
    
    # Sort upper points from left to right
    upper.sort(key=lambda p: p[0])
    
    # Sort lower points from left to right
    lower.sort(key=lambda p: p[0])
    
    # Combine in the desired order: upper left, upper right, lower right, lower left
    ordered_corners = [upper[0], upper[1], lower[1], lower[0]]
    
    # If original had Z coordinates, add them back (using the original Z values)
    if has_z:
        z_values = coords[:, 2]
        ordered_corners_3d = [
            (corner[0], corner[1], z_values[np.where((coords[:, :2] == corner).all(axis=1))[0][0]])
            for corner in ordered_corners
        ]
        # Create new polygon with ordered corners (including Z)
        return Polygon(ordered_corners_3d + [ordered_corners_3d[0]])
    else:
        # Create new polygon with ordered corners
        return Polygon(ordered_corners + [ordered_corners[0]])


def yolo_to_projected_polygon(image_coords, yolo_coords):
    """
    Convert YOLO format bounding box coordinates to a projected polygon, considering yaw.

    Parameters:
    image_coords (list of tuples): List of 4 tuples representing the coordinates
                                   of the image corners in UTM projection,
                                   in order: upper left, upper right, lower right, lower left.
    yolo_coords (tuple): Tuple containing the YOLO format bounding box coordinates
                         in the order: (x_center, y_center, width, height)
                         These should be normalized values between 0 and 1.
    yaw (float): Yaw angle in degrees, representing the flight direction (not used in this version).

    Returns:
    shapely.geometry.Polygon: A Polygon representing the bounding box in the
                              projected coordinate system.
    """
    # Unpack image coordinates
    ul, ur, lr, ll = image_coords

    # Create vectors for width and height of the image in projected space
    width_vector = np.array(ur) - np.array(ul)  # Vector from UL to UR (image width)
    height_vector = np.array(ll) - np.array(ul)  # Vector from UL to LL (image height)

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
    geom_out = yolo_to_projected_polygon(
        image_coords, yolo_coords
    )
    return geom_out


def get_best_utm_epsg(gdf):
    """
    Determine the best UTM zone EPSG code for a GeoDataFrame in EPSG:4326.

    Parameters:
        gdf (geopandas.GeoDataFrame): A GeoDataFrame with a CRS of EPSG:4326.

    Returns:
        int: The EPSG code for the best matching UTM zone.
    """
    # Ensure the GeoDataFrame is in EPSG:4326
    if gdf.crs.to_epsg() != 4326:
        raise ValueError("GeoDataFrame must be in EPSG:4326 projection.")

    # Calculate the centroid of the bounding box
    bbox = gdf.total_bounds  # [minx, miny, maxx, maxy]
    centroid_lon = (bbox[0] + bbox[2]) / 2  # Average longitude
    centroid_lat = (bbox[1] + bbox[3]) / 2  # Average latitude

    # Determine UTM zone based on longitude
    utm_zone = int((centroid_lon + 180) // 6) + 1

    # Determine if it's northern or southern hemisphere
    if centroid_lat >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere

    return epsg_code
