import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def validate_sides(coords):
    return Point(coords[0]).distance(Point(coords[1])) > Point(coords[1]).distance(
        Point(coords[2])
    )


def sort_vertices(coords: np.array, yaw: float, tolerance=3, verbose: bool = False):
    """
    Sort coordinates of a polygon based on the yaw angle.

    This function identifies the upper-left corner of a polygon based on the yaw angle,
    and then sorts the coordinates in a counter-clockwise order starting from that corner.

    Args:
        coords (numpy.ndarray): Array of coordinates representing the vertices of a polygon.
        yaw (float): Yaw angle in degrees, representing the orientation of the polygon.

    Returns:
        numpy.ndarray: Sorted coordinates of the polygon.
    """
    yaw = float(yaw)
    if yaw >= 0 + tolerance and yaw < 90 - tolerance:
        ul = coords[:, 1].argmax()
        ur = coords[:, 0].argmax()
        lr = coords[:, 1].argmin()
        ll = coords[:, 0].argmin()
    elif yaw >= 90 + tolerance and yaw < 180 - tolerance:
        ul = coords[:, 0].argmax()
        ur = coords[:, 1].argmin()
        lr = coords[:, 0].argmin()
        ll = coords[:, 1].argmax()
    elif yaw >= 180 + tolerance and yaw < 270 - tolerance:
        ul = coords[:, 1].argmin()
        ur = coords[:, 0].argmin()
        lr = coords[:, 1].argmax()
        ll = coords[:, 0].argmax()
    elif yaw >= 270 + tolerance and yaw <= 360 - tolerance:
        ul = coords[:, 0].argmin()
        ur = coords[:, 1].argmax()
        lr = coords[:, 0].argmax()
        ll = coords[:, 1].argmin()
    elif yaw >= 360 - tolerance or yaw < 0 + tolerance:
        df = pd.DataFrame(data=coords, columns=["x", "y", "z"]).sort_values("y")
        upper = df.iloc[:2].sort_values("x")
        lower = df.iloc[2:].sort_values("x", ascending=False)
        ul, ur, lr, ll = pd.concat([upper, lower]).index.values
    elif yaw >= 90 - tolerance and yaw < 90 + tolerance:
        df = pd.DataFrame(data=coords, columns=["x", "y", "z"]).sort_values(
            "x", ascending=False
        )
        right = df.iloc[:2].sort_values("y", ascending=False)
        left = df.iloc[2:].sort_values("y")
        ul, ur, lr, ll = pd.concat([right, left]).index.values
    elif yaw >= 180 - tolerance and yaw < 180 + tolerance:
        df = pd.DataFrame(data=coords, columns=["x", "y", "z"]).sort_values("y")
        upper = df.iloc[:2].sort_values("x")
        lower = df.iloc[2:].sort_values("x", ascending=False)
        ul, ur, lr, ll = pd.concat([lower, upper]).index.values
    elif yaw >= 270 - tolerance and yaw < 270 + tolerance:
        df = pd.DataFrame(data=coords, columns=["x", "y", "z"]).sort_values("x")
        left = df.iloc[:2].sort_values("y")
        right = df.iloc[2:].sort_values("y", ascending=False)
        ul, ur, lr, ll = pd.concat([left, right]).index.values
    else:
        raise ValueError("Yaw angle out of range")

    order = [ul, ur, lr, ll]
    if verbose:
        print(order)

    new_coords = coords[order]

    valid_direction = validate_sides(new_coords)
    if verbose:
        print(validate_sides(valid_direction))
    return new_coords


def correct_coords(coords, yaw: float):
    new_coords = sort_vertices(coords, yaw)
    if validate_sides:
        return new_coords
    else:
        new_coords = sort_vertices(coords, yaw + 3)
        if validate_sides:
            return new_coords
        else:
            new_coords = sort_vertices(coords, yaw - 3)
            if validate_sides:
                return new_coords
            else:
                return coords


def set_footprint(row):
    """
    Creates a Shapely Polygon object from a row of data, reordering the coordinates.

    The function extracts coordinates and yaw angle from the input row,
    then uses the `manual_sort` function to reorder the coordinates
    based on the yaw angle, and finally creates a Polygon object.

    Args:
        row (pandas.Series): A row of data containing 'geometry' (Shapely Polygon) and 'Yaw[deg]' columns.

    Returns:
        shapely.geometry.Polygon: A Polygon object with sorted coordinates.
    """
    polygon = row.geometry
    coords = np.array(list(polygon.exterior.coords)[:4])
    yaw = row["Yaw[deg]"]
    # coords_new = manual_sort(coords, yaw=yaw)
    coords_new = correct_coords(coords, yaw=yaw)
    return Polygon(coords_new)
