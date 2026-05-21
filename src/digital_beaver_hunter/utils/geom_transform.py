from typing import Sequence, Tuple
import numpy as np
from shapely.geometry import Polygon


def get_polygon_corners(poly: Polygon):
    coords = list(poly.exterior.coords)[:-1]  # drop the repeated first point
    # assume order is ul, ur, lr, ll or similar; you may need to sort by your convention
    return np.array(coords)[:, :2]  # shape (4, 2): [E, N]


def _wrap_360(angle_deg: np.ndarray) -> np.ndarray:
    """Wrap angles to the interval [0, 360)."""
    return angle_deg % 360.0


def sort_corners_by_flight(
    corners: np.ndarray,
    yaw_deg: float,
) -> np.ndarray:
    """
    Re‑order a (4, 2) array of corner coordinates so that they are returned
    as [UL, UR, LR, LL] *relative to the aircraft heading*.

    Parameters
    ----------
    corners : np.ndarray
        Shape (4, 2) – the four corner points in the same CRS (easting, northing).
    yaw_deg : float
        Aircraft heading (yaw) in **degrees**, measured clockwise from true north
        (0 ° = north, 90 ° = east).

    Returns
    -------
    np.ndarray
        Re‑ordered corners (shape (4, 2)) in the order UL → UR → LR → LL.

    Notes
    -----
    The algorithm:
        1. Compute the centroid of the four points.
        2. For every corner compute the bearing from the centroid
           (atan2(easting‑centroid, northing‑centroid) → degrees, 0 ° = north).
        3. Subtract the aircraft yaw so that “forward” aligns with 0 °.
        4. Wrap the resulting angle to [0, 360) and place each corner in one
           of four 90‑degree sectors:
                0°‑90°   → front‑right   (UR)
                90°‑180° → rear‑right    (LR)
                180°‑270°→ rear‑left     (LL)
                270°‑360°→ front‑left    (UL)
        5. Return the corners in UL, UR, LR, LL order.
    """
    # --------------------------------------------------------------
    # 0️⃣ Basic sanity check
    # --------------------------------------------------------------
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (4, 2):
        raise ValueError(
            "corners must be an array‑like of shape (4, 2) "
            f"(UL, UR, LR, LL). Got shape {corners.shape}."
        )

    # --------------------------------------------------------------
    # 1️⃣ Image centre (centroid)
    # --------------------------------------------------------------
    centroid = corners.mean(axis=0)  # (easting_c, northing_c)

    # --------------------------------------------------------------
    # 2️⃣ Bearing of each corner w.r.t. the centroid
    #    atan2 expects (dx, dy) → (easting‑centroid, northing‑centroid)
    #    The result is in radians, 0 rad = north, positive clockwise.
    # --------------------------------------------------------------
    dx = corners[:, 0] - centroid[0]
    dy = corners[:, 1] - centroid[1]
    bearing_rad = np.arctan2(dx, dy)  # note the order (dx, dy)
    bearing_deg = np.rad2deg(bearing_rad)  # 0‑360 (but may be negative)

    # --------------------------------------------------------------
    # 3️⃣ Rotate the bearings by the aircraft heading so that
    #    “forward” = 0 ° in the aircraft‑reference frame.
    # --------------------------------------------------------------
    rel_angle = _wrap_360(bearing_deg - yaw_deg)  # now 0‑360

    # --------------------------------------------------------------
    # 4️⃣ Assign each corner to a sector.
    #    The sector limits are inclusive on the lower bound, exclusive on the upper.
    # --------------------------------------------------------------
    #   0°‑90°   → front‑right → UR
    #   90°‑180° → rear‑right  → LR
    #   180°‑270°→ rear‑left   → LL
    #   270°‑360°→ front‑left  → UL
    sector = np.empty(4, dtype=int)

    sector[(0 <= rel_angle) & (rel_angle < 90)] = 1  # UR
    sector[(90 <= rel_angle) & (rel_angle < 180)] = 2  # LR
    sector[(180 <= rel_angle) & (rel_angle < 270)] = 3  # LL
    sector[(270 <= rel_angle) & (rel_angle < 360)] = 0  # UL

    # --------------------------------------------------------------
    # 5️⃣ Build the output array in the required order.
    # --------------------------------------------------------------
    #   order = [UL, UR, LR, LL]  →  sector indices [0, 1, 2, 3]
    ordered_idx = np.empty(4, dtype=int)
    for target, sec_id in enumerate([0, 1, 2, 3]):  # UL,UR,LR,LL
        # there must be exactly one corner per sector; otherwise the geometry is
        # degenerate (e.g., two corners share the same sector because the image
        # is a line or all four points are collinear).
        matches = np.where(sector == sec_id)[0]
        if matches.size != 1:
            raise RuntimeError(
                f"Unable to uniquely assign a corner to sector {sec_id} "
                f"(found {matches.size} matches).  Check the input geometry."
            )
        ordered_idx[target] = matches[0]

    return corners[ordered_idx]


def rel_to_proj(rx: float, ry: float, corners_ul_ur_lr_ll: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation inside a quadrilateral.

    Parameters
    ----------
    rx, ry : float
        Relative coordinates in the range [0, 1].
        *rx* runs from left (0) → right (1) across the image.
        *ry* runs from top   (0) → bottom (1) down the image.
    corners_ul_ur_lr_ll : np.ndarray
        Array of shape (4, 2) ordered as
        [UL, UR, LR, LL] in the same CRS as the output.

    Returns
    -------
    np.ndarray
        (2,) array → [easting, northing] of the projected point.
    """
    # Unpack the corners for readability
    ul, ur, lr, ll = corners_ul_ur_lr_ll

    # Interpolate along the top and bottom edges first
    top = (1 - rx) * ul + rx * ur  # point on the top edge (ry = 0)
    bottom = (1 - rx) * ll + rx * lr  # point on the bottom edge (ry = 1)

    # Then interpolate between those two points vertically
    proj = (1 - ry) * top + ry * bottom
    return proj


def make_footprint_polygon(
    corners: Sequence[Sequence[float]],
    *,
    bbox: bool = False,
) -> Polygon | Tuple[Polygon, Polygon]:
    """
    Build a Shapely Polygon representing the footprint of an aerial image.

    Parameters
    ----------
    corners : array‑like of shape (4, 2)
        Ordered corner coordinates **[UL, UR, LR, LL]** in the same CRS you will
        later assign to the geometry.  They can be a list of tuples, a NumPy
        array, etc.
    bbox : bool, default ``False``
        If ``True`` also create the axis‑aligned bounding‑box rectangle.
        The function then returns ``(quadrilateral, bounding_box)``.
        If ``False`` only the quadrilateral polygon is returned.

    Returns
    -------
    shapely.geometry.Polygon
        The exact quadrilateral defined by the four corners, **or**
    tuple(Polygon, Polygon)
        ``(quad, bbox)`` when ``bbox=True``.

    Raises
    ------
    ValueError
        If *corners* is not of shape (4, 2).

    Example
    -------
    >>> corners = [
    ...     (606968.13345377, 7476395.35925286),   # UL
    ...     (607265.94948087, 7475422.35325698),   # UR
    ...     (605887.58533231, 7475050.717058  ),   # LR
    ...     (605622.40075732, 7476022.13632393)    # LL
    ... ]
    >>> quad = make_footprint_polygon(corners)
    >>> quad.wkt
    'POLYGON ((606968.13345377 7476395.35925286, 607265.94948087 7475422.35325698,
               605887.58533231 7475050.717058, 605622.40075732 7476022.13632393,
               606968.13345377 7476395.35925286))'
    >>> quad, rect = make_footprint_polygon(corners, bbox=True)   # get both
    """
    # ------------------------------------------------------------------
    # 1️⃣ Validate the input and convert to a NumPy array (fast indexing)
    # ------------------------------------------------------------------
    arr = np.asarray(corners, dtype=float)
    if arr.shape != (4, 2):
        raise ValueError(
            "corners must be an array‑like of shape (4, 2) representing "
            f"UL, UR, LR, LL.  Got shape {arr.shape}."
        )

    # ------------------------------------------------------------------
    # 2️⃣ Build the exact quadrilateral polygon
    # ------------------------------------------------------------------
    # Shapely automatically closes the ring, but we repeat the first point
    # for clarity (makes the WKT easier to read).
    quad_coords = np.vstack([arr, arr[0]])  # (5,2)
    quad = Polygon(quad_coords)

    # ------------------------------------------------------------------
    # 3️⃣ (Optional) Build the axis‑aligned bounding‑box rectangle
    # ------------------------------------------------------------------
    if bbox:
        xmin, ymin, xmax, ymax = quad.bounds
        bbox_coords = [
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin),
            (xmin, ymin),
            (xmin, ymax),
        ]
        rect = Polygon(bbox_coords)
        return quad, rect

    return quad
