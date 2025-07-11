import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle, Circle, Wedge, Polygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import erf
from scipy.optimize import minimize
from numpy.random import default_rng
import xml.etree.ElementTree as ET
import pandas as pd
from shapely.geometry import Point, Polygon as ShapelyPolygon
from shapely.geometry import MultiPoint
from shapely.prepared import prep
import re

rng = default_rng()


class DetectorParams:
    """All detector and electronics parameters in one place"""

    # Gas properties / drift
    D_T = 175.0  # μm/√cm - transverse diffusion coefficient
    D_L = 230.0  # μm/√cm - longitudinal diffusion coefficient
    v_drift = 8.0e-3  # cm/ns = 8.0e-3 cm/ns - drift velocity
    TPC_length = 105.5  # cm - TPC drift length (cathode to readout)

    # Avalanche intrinsic spreads (GEM/Micromegas)
    sigma_aval_r = 1.5  # mm - transverse spread from avalanche
    sigma_aval_t = sigma_aval_r / v_drift / 10  # ns - temporal spread from avalanche

    # Gain parameters
    gain_mean = 1400  # Average avalanche gain
    polya_theta = 0.8  # Polya parameter

    # Electronics shaping (CR-RC with n=1)
    tau_shaper = 50.0  # ns - shaping time constant
    # sampa_shaping_lead = 32.0  # ns - SAMPA shaping time lead
    # sampa_shaping_tail = 48.0  # ns - SAMPA shaping time tail

    # Digitization
    adc_dt = 53.326184  # ns - TPC ADC clock period
    window_ns = 22653.0  # ns - total readout window (425 samples)
    adc_bits = 10
    adc_conversion = 0.1  # mV per electron (after gain)
    adc_noise = 5.0  # ADC counts RMS noise
    adc_threshold = 0  # ADC threshold

    # Pad geometry
    pad_geometry = "rectangular"  # 'rectangular' or 'rphi'

    # For rectangular pads
    pad_width_mm = 4.0  # mm - pad width in x
    pad_height_mm = 4.0  # mm - pad height in y

    # For r-φ pads
    pad_r_mm = 4.0  # mm - pad radial size
    pad_phi_mrad = 10.0  # mrad - pad angular size (at reference radius)
    r_reference = 100.0  # mm - reference radius for phi calculation

    # Coverage
    n_sigma_pad = 3


NSectors = 12
min_radii_module = [314.9836110818037, 416.59202613529567, 589.1096495597712]
max_radii_module = [399.85222874031024, 569.695373910603, 753.6667758418596]
delta_R_module = [
    5.630547309825637,
    5.6891770257002054,
    10.206889851687158,
    10.970475085472556,
]
left_phi_module = [1.814070180963218, 1.8196038006425466, 1.8231920135309803]
right_phi_module = [1.3233741362052793, 1.3190806892129499, 1.3183480511410641]
tpc_x_sizes = [[-101, 101], [-151, 151], [-201, 201]]
tpc_y_sizes = [[300, 405], [395, 575], [560, 760]]
# Global derived values
P = DetectorParams()
NT = int(P.window_ns // P.adc_dt)
BIN_EDGES = np.arange(NT + 1) * P.adc_dt
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])


def rotate_point_to_sector(x, y):
    """
    Given x,y in mm, find the TPC wedge (of 12) that (x,y) falls into,
    then rotate the point so that that wedge is moved to the 12-o'clock
    position (i.e. centered at phi = +90°).

    Returns:
      x_new, y_new : the rotated coordinates (still in mm)
      sector       : the original wedge index [0..11]
    """
    # 1) compute original phi
    phi = np.arctan2(y, x)  # in [−π, +π]

    # 2) figure out which 30°‐wide wedge it lives in
    #    we want wedge #0 centered at 0° (3 o'clock), wedge #1 at 30°, …, wedge #3 at 90° (12 o'clock), etc.
    wedge_width = 2 * np.pi / NSectors  # = π/6
    # shift phi by + half‐wedge so that floor() bins correctly
    sector = int(np.floor((phi + wedge_width / 2) / wedge_width)) % NSectors

    # 3) compute how much to rotate so that that sector’s center (sector * 30°)
    #    is brought to +90° (= π/2)
    target_center = np.pi / 2  # 12 o'clock
    original_center = (
        sector * wedge_width
    )  # e.g. sector = 0 → 0 rad, sector = 3 → π/2 rad, etc.
    dphi = target_center - original_center

    # 4) apply rotation
    R = np.hypot(x, y)
    phi_rot = phi + dphi
    x_new = R * np.cos(phi_rot)
    y_new = R * np.sin(phi_rot)

    return x_new, y_new, sector


'''def rotate_point_to_sector(x0, y0, z0):
    """
    Rotate a single point (x0,y0,z0) in mm into the sector-0 wedge.
    Returns: x_rot, y_rot, z0, sector, side
    """
    # 1) radius and phi
    R = np.hypot(x0, y0)
    if R > max_radii_module[2] + 1e-6:
        raise ValueError(f"Point R={R:.1f} mm outside outer radius")
    phi = np.arctan2(y0, x0)

    # 2) which side of the TPC (z<0 => side=1)
    side = int(z0 < 0)

    # 3) determine the sector index
    pie = np.pi / NSectors  # wedge half-width
    raw = phi - 2 * np.pi / 6  # align your "2π/6" offset
    sector = int(abs((raw // pie)))  # floor division then abs

    # 4) compute the shift to bring that sector back to “sector 0”
    phi_shift = np.pi - (sector + 2) * np.pi / 6

    # 5) rotate the point
    phi0 = phi - phi_shift
    x_rot = R * np.cos(phi0)
    y_rot = R * np.sin(phi0)

    return x_rot, y_rot, z0, sector, side'''


def build_serf_zigzag(x0_cm, y0_cm, z0_cm, df_centroids, return_components=False):
    """
    Build Single Electron Response Function for zigzag pads

    Args:
        x0_cm, y0_cm, z0_cm: Initial electron position in cm
        df_centroids: List of DataFrames for each module
        return_components: If True, return intermediate arrays
    """
    # Convert to mm
    x0_mm = x0_cm * 10
    y0_mm = y0_cm * 10
    z0_mm = z0_cm * 10

    # Calculate cloud parameters
    cloud_params = calculate_cloud_parameters(z0_cm)

    # Get pad charge fractions
    pad_fractions, module = get_zigzag_pad_fractions(
        x0_mm, y0_mm, z0_mm, cloud_params["sigma_total_r"], df_centroids
    )

    # Time distribution (before shaping)
    time_fractions = np.zeros(NT)
    for k in range(NT):
        time_fractions[k] = gaussian_integral_1d(
            BIN_EDGES[k],
            BIN_EDGES[k + 1],
            cloud_params["t_drift_ns"],
            cloud_params["sigma_total_t"],
        )

    # Apply shaping
    shaped_response = np.convolve(time_fractions, SHAPER_KERNEL, mode="full")[:NT]

    # Normalize
    if np.sum(shaped_response) > 0:
        shaped_response = shaped_response / np.sum(shaped_response)

    # Build SERF
    serf = {}
    for pad_id, frac in pad_fractions.items():
        serf[(module, pad_id)] = frac * shaped_response

    # Prepare return values
    params = {
        **cloud_params,
        "x0_mm": x0_mm,
        "y0_mm": y0_mm,
        "module": module,
        "n_pads": len(pad_fractions),
    }

    if return_components:
        components = {
            "pad_fractions": pad_fractions,
            "time_fractions": time_fractions,
            "shaped_response": shaped_response,
        }
        return serf, params, components
    else:
        return serf, params


def build_serf_zigzag_in_sector(
    x0_cm, y0_cm, z0_cm, df_centroids, return_components=False
):
    """
    Rotate (x0,y0,z0) into sector 0 and then call existing zigzag SERF.
    x0_cm,y0_cm in cm, z0_cm in cm.
    """
    # prepare “track” in mm
    # track = {
    #  'x': [x0_cm*10.0],
    #  'y': [y0_cm*10.0],
    #  'z': [z0_cm*10.0]
    # }
    x0_mm = x0_cm * 10.0
    y0_mm = y0_cm * 10.0
    z0_mm = z0_cm * 10.0  # convert to mm for rotation
    ti = rotate_point_to_sector(x0_mm, y0_mm)
    # unpack the one (and only) rotated point, back to cm
    x0s = ti[0] / 10.0
    y0s = ti[1] / 10.0

    # now call your old zigzag SERF
    serf, params, comps = build_serf_zigzag(
        x0s, y0s, z0_cm, df_centroids, return_components=True
    )

    # tag on sector/side if you want to plot it
    # params["sector"] = int(ti[3])
    # params["side"] = int(ti[4])
    params["sector"] = 0
    params["side"] = 1
    if return_components:
        return serf, params, comps
    else:
        return serf, params


maps = [
    "/Users/chenxi/developer/sphenix/tpc_sim/TPC_Toy_Model/PadPlane/AutoPad-R1-RevA.sch",
    "/Users/chenxi/developer/sphenix/tpc_sim/TPC_Toy_Model/PadPlane/AutoPad-R2-RevA-Pads.sch",
    "/Users/chenxi/developer/sphenix/tpc_sim/TPC_Toy_Model/PadPlane/AutoPad-R3-RevA.sch",
]


def get_pad_coordinates(root):
    """Reading all the vertices of the pads from the xml ElementTree"""
    all_elements = [elem.tag for elem in root.iter()]
    all_attributes = [elem.attrib for elem in root.iter()]
    i = 0
    f_vertex = False
    f_wire = False
    all_vertices = []
    vertex_array = []
    pad_names = []
    pad_name = ""

    for ele in all_elements:
        if f_vertex:
            if ele == "vertex":
                vertex_array.append(all_attributes[i])
            else:
                pad_names.append(pad_name)
                all_vertices.append(vertex_array)
                f_vertex = False
                f_wire = False
        if f_wire:
            if ele == "wire" and all_attributes[i]["layer"] == "16":
                vertex_array.append(
                    {"x": all_attributes[i]["x1"], "y": all_attributes[i]["y1"]}
                )
                vertex_array.append(
                    {"x": all_attributes[i]["x2"], "y": all_attributes[i]["y2"]}
                )
            elif ele == "signal":
                pad_names.append(pad_name)
                all_vertices.append(vertex_array)
                f_vertex = False
                f_wire = False

        if ele == "signal":
            pad_name = all_attributes[i]["name"]

        if ele == "polygon":
            if all_attributes[i]["layer"] == "16":
                f_vertex = True
                vertex_array = []

        if re.match(r"DC*", pad_name):
            if ele == "signal":
                f_wire = True
                vertex_array = []

        i += 1

    return all_vertices, pad_names


def process_pad_vertices(all_vertices, all_PadNames):
    """Process pad vertices and create DataFrame with pad properties"""
    pads_xy = {
        "PadName": [],
        "PadX": [],
        "PadY": [],
        "PadR": [],
        "PadPhi": [],
        "PadPath": [],
        "PadNumber": [],
    }

    iter = 0
    for vx_array, padName in zip(all_vertices, all_PadNames):
        iter += 1

        # Extract vertices
        x_vx = []
        y_vx = []
        r_vx = []
        phi_vx = []
        xy_vx = []

        for vx in vx_array:
            x_vx.append(float(vx["x"]))
            y_vx.append(float(vx["y"]))
            r = np.sqrt(float(vx["y"]) ** 2 + float(vx["x"]) ** 2)
            phi = np.arctan2(float(vx["y"]), float(vx["x"]))
            r_vx.append(r)
            phi_vx.append(phi)
            xy_vx.append([float(vx["x"]), float(vx["y"])])

        # Calculate centroid
        phi_vx.sort()
        phi_mins = phi_vx[:18] if len(phi_vx) >= 18 else phi_vx[: len(phi_vx) // 2]
        phi_maxs = phi_vx[-18:] if len(phi_vx) >= 18 else phi_vx[len(phi_vx) // 2 :]
        new_phi_min = sum(phi_mins) / len(phi_mins) if phi_mins else phi_vx[0]
        new_phi_max = sum(phi_maxs) / len(phi_maxs) if phi_maxs else phi_vx[-1]

        # Make Path
        xy_vx.append(xy_vx[0])  # Close the polygon
        codes = [Path.MOVETO] + [Path.LINETO] * (len(xy_vx) - 2) + [Path.CLOSEPOLY]
        path = Path(xy_vx, codes)

        # Calculate center
        r_vx.sort()
        if len(r_vx) > 4:
            r_vx = r_vx[2:-2]
        c_r = sum(r_vx) / len(r_vx) if r_vx else 0
        c_phi = (new_phi_max + new_phi_min) / 2
        c_x = c_r * np.cos(c_phi)
        c_y = c_r * np.sin(c_phi)

        pads_xy["PadName"].append(padName)
        pads_xy["PadX"].append(c_x)
        pads_xy["PadY"].append(c_y)
        pads_xy["PadR"].append(c_r)
        pads_xy["PadPhi"].append(c_phi)
        pads_xy["PadPath"].append(path)
        pads_xy["PadNumber"].append(iter)

    df = pd.DataFrame(data=pads_xy)

    # Sort by radius and phi
    df["PadR_group"] = df["PadR"].round(3)
    df_sorted = df.sort_values(by=["PadR_group", "PadPhi"], ascending=[True, False])
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted["PadNumber"] = df_sorted.index + 1

    return df_sorted


def load_pad_planes(brd_maps):
    """Load all three pad plane modules"""
    df_centroids = []

    for module_idx, brd_map in enumerate(brd_maps):
        tree = ET.parse(brd_map)
        root = tree.getroot()

        # after: give it the Element
        all_vertices, all_PadNames = get_pad_coordinates(tree)

        # Process into DataFrame
        df_module = process_pad_vertices(all_vertices, all_PadNames)
        df_centroids.append(df_module)

        print(f"Module {module_idx}: Loaded {len(df_module)} pads")

    return df_centroids


def find_pad_for_point(x, y, df):
    """Find which pad contains a given point"""
    point = (x, y)
    for _, row in df.iterrows():
        pad_path = row["PadPath"]
        if pad_path.contains_point(point):
            return row["PadNumber"]
    return None


def edge_pads_hit(hit_x, hit_y, cloud_r, df, num_points=20):
    """Find all pads touched by electron cloud"""
    hit_pads = set()

    # Check center point
    center_pad = find_pad_for_point(hit_x, hit_y, df)
    if center_pad is not None:
        hit_pads.add(center_pad)

    # Check points on circumference
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    for theta in angles:
        x = hit_x + cloud_r * np.cos(theta)
        y = hit_y + cloud_r * np.sin(theta)
        pad = find_pad_for_point(x, y, df)
        if pad is not None:
            hit_pads.add(pad)

    return list(hit_pads)


def integrated_density_of_circle_and_pad(hit_x, hit_y, sigma, pad_path, grid_step=None):
    """Calculate charge fraction on a pad from 2D Gaussian cloud"""

    # Convert Path to shapely Polygon
    if hasattr(pad_path, "vertices"):
        pad_coords = pad_path.vertices
    else:
        pad_coords = pad_path

    pad = ShapelyPolygon(pad_coords)
    if not pad.is_valid:
        pad = pad.buffer(0)

    # Get intersection with circular cloud (n-sigma radius)
    n_sigma = 3
    circle = Point(hit_x, hit_y).buffer(n_sigma * sigma)
    intersection = circle.intersection(pad)

    if intersection.is_empty:
        return 0.0

    # Set grid resolution
    if grid_step is None:
        grid_step = sigma / 50.0  # Fine grid for accuracy

    # Define Gaussian
    gaussian_constant = 1 / (2 * np.pi * sigma**2)
    exp_denominator = 2 * sigma**2

    # Get bounding box
    minx, miny, maxx, maxy = intersection.bounds
    xs = np.arange(minx, maxx, grid_step)
    ys = np.arange(miny, maxy, grid_step)

    if len(xs) == 0 or len(ys) == 0:
        return 0.0

    # Create meshgrid
    xx, yy = np.meshgrid(xs, ys)
    points = np.vstack((xx.ravel(), yy.ravel())).T

    # Check which points are inside
    prepared_intersection = prep(intersection)
    mask = np.array([prepared_intersection.contains(Point(p)) for p in points])
    inside_points = points[mask]

    if len(inside_points) == 0:
        return 0.0

    # Calculate Gaussian values
    r2 = (inside_points[:, 0] - hit_x) ** 2 + (inside_points[:, 1] - hit_y) ** 2
    densities = gaussian_constant * np.exp(-r2 / exp_denominator)

    # Integrate
    total_density = np.sum(densities) * (grid_step**2)

    return total_density


def which_layer(r):
    """Determine module and layer from radius"""
    # Module boundaries (in mm)
    min_radii_module = [314.9836, 416.5920, 589.1096]
    max_radii_module = [399.8522, 569.6954, 753.6668]
    delta_R_module = [5.6305, 5.6892, 10.2069, 10.9705]

    module = -1
    layer = -1

    # Find module
    for imodule in range(3):
        if r >= min_radii_module[imodule] and r < max_radii_module[imodule]:
            module = imodule
            break

    if module == -1:
        return -1, -1

    Nlayers = 16

    # Find layer within module
    if module == 0:
        for iraw in range(Nlayers):
            n_pads_small = int((iraw + 1) / 2)
            n_pads_large = int(iraw / 2)
            radius = (
                min_radii_module[module]
                + delta_R_module[module] * n_pads_small
                + delta_R_module[module + 1] * n_pads_large
            )
            if r > radius and r < radius + delta_R_module[module + (iraw % 2)]:
                layer = iraw
                break
    else:
        for iraw in range(Nlayers):
            radius = min_radii_module[module] + delta_R_module[module + 1] * iraw
            if r > radius and r < radius + delta_R_module[module + 1]:
                layer = iraw
                break

    return module, layer


def compute_shaper_kernel():
    """CR-RC semi-Gaussian response, normalized to unit area"""
    t = BIN_CENTERS
    h = (t / P.tau_shaper**2) * np.exp(-t / P.tau_shaper)
    h[t < 0] = 0
    # Normalize
    h = h / (np.sum(h) * P.adc_dt)
    return h


SHAPER_KERNEL = compute_shaper_kernel()


def gaussian_integral_1d(a, b, mu, sigma):
    """Analytic integral of 1D Gaussian from a to b"""
    if sigma <= 0:
        return 0.0
    return 0.5 * (
        erf((b - mu) / (np.sqrt(2) * sigma)) - erf((a - mu) / (np.sqrt(2) * sigma))
    )


def calculate_cloud_parameters(z0_cm):
    """
    Calculate expected cloud parameters from drift distance
    Combines drift diffusion with avalanche spread
    """
    # Drift distance
    dz_cm = abs(z0_cm - P.TPC_length)

    # Drift diffusion spreads
    sigma_drift_r_mm = P.D_T * np.sqrt(dz_cm) / 1000.0  # μm to mm
    sigma_drift_t_ns = (P.D_L * np.sqrt(dz_cm) / 1e4) / P.v_drift  # to ns

    # Combine with avalanche spreads (add in quadrature)
    sigma_r_total = np.hypot(sigma_drift_r_mm, P.sigma_aval_r)
    sigma_t_total = np.hypot(sigma_drift_t_ns, P.sigma_aval_t)

    # Mean drift time
    t_drift_ns = dz_cm / P.v_drift

    return {
        "dz_cm": dz_cm,
        "sigma_drift_r": sigma_drift_r_mm,
        "sigma_drift_t": sigma_drift_t_ns,
        "sigma_total_r": sigma_r_total,
        "sigma_total_t": sigma_t_total,
        "t_drift_ns": t_drift_ns,
    }


def get_zigzag_pad_fractions(x0_mm, y0_mm, z0_mm, sigma_r, df_centroids):
    """Get pad charge fractions for zigzag geometry"""

    # Determine which module
    r0 = np.sqrt(x0_mm**2 + y0_mm**2)
    module, layer = which_layer(r0)

    if module == -1:
        return {}, module

    df = df_centroids[module]

    # Find hit pads
    hit_pads = edge_pads_hit(x0_mm, y0_mm, sigma_r, df, num_points=100)

    # Calculate charge fraction for each pad
    pad_fractions = {}
    for pad_id in hit_pads:
        pad_row = df.loc[df["PadNumber"] == pad_id].iloc[0]
        pad_path = pad_row["PadPath"]

        fraction = integrated_density_of_circle_and_pad(x0_mm, y0_mm, sigma_r, pad_path)

        if fraction > 1e-8:
            pad_fractions[pad_id] = fraction

    # Normalize
    total = sum(pad_fractions.values())
    if total > 0:
        pad_fractions = {k: v / total for k, v in pad_fractions.items()}

    return pad_fractions, module


def polya_gain(mean=P.gain_mean, theta=P.polya_theta):
    """Sample from Polya distribution for realistic gain fluctuations"""
    return rng.gamma(shape=theta, scale=mean / theta)


def exponential_gain(mean=P.gain_mean):
    """Alternative: exponential gain distribution"""
    return rng.exponential(scale=mean)


def digitize_serf(serf, gain=None, add_noise=True):
    """
    Convert SERF to digitized ADC values

    Args:
        serf: Dict from build_serf
        gain: Avalanche gain (if None, sample from Polya)
        add_noise: Whether to add electronic noise

    Returns:
        Dict of pad_id -> ADC array
    """
    if gain is None:
        gain = polya_gain()

    waveforms = {}
    for pad_id, response in serf.items():
        # Convert to voltage (arbitrary units)
        adc = gain * response * P.adc_conversion

        # Add noise
        if add_noise:
            noise = rng.normal(0, P.adc_noise, len(response))
            adc = adc + noise

        # Digitize and clip
        adc = np.clip(np.round(adc), 0, 2**P.adc_bits - 1).astype(int)

        # Store if above threshold
        # if np.max(adc) > P.adc_threshold:
        waveforms[pad_id] = adc

    return waveforms


def visualize_complete_response_zigzag(x0, y0, z0, df_centroids):
    """Comprehensive visualization for zigzag geometry"""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Build SERF with components
    serf, params, components = build_serf_zigzag_in_sector(
        x0, y0, z0, df_centroids, return_components=True
    )

    # 1. Pad charge distribution
    ax1 = fig.add_subplot(gs[0, 0])

    if params["module"] >= 0:
        df = df_centroids[params["module"]]
        pad_fracs = components["pad_fractions"]

        # Draw all pads in region
        r0 = np.sqrt(params["x0_mm"] ** 2 + params["y0_mm"] ** 2)

        # Find pads to draw
        pads_to_draw = df[(df["PadR"] > r0 - 50) & (df["PadR"] < r0 + 50)]

        # Draw pads
        patches = []
        colors = []

        for _, row in pads_to_draw.iterrows():
            pad_path = row["PadPath"]
            vertices = pad_path.vertices
            polygon = Polygon(vertices, closed=True)
            patches.append(polygon)

            # Color by charge fraction
            if row["PadNumber"] in pad_fracs:
                colors.append(pad_fracs[row["PadNumber"]])
            else:
                colors.append(0)

        pc = PatchCollection(patches, cmap="hot", edgecolors="gray", linewidths=0.5)
        pc.set_array(np.array(colors))
        ax1.add_collection(pc)

        # Add cloud outline
        circle = Circle(
            (params["x0_mm"], params["y0_mm"]),
            params["sigma_total_r"],
            fill=False,
            color="blue",
            linewidth=2,
        )
        ax1.add_patch(circle)
        # ax1.plot(params["x0_mm"], params["y0_mm"], "b*", markersize=10)
        # Overlay 2D Gaussian heatmap of the electron cloud
        sigma = params["sigma_total_r"]
        x0 = params["x0_mm"]
        y0 = params["y0_mm"]
        grid_extent = 3 * sigma
        n = 100  # resolution of the heatmap
        x = np.linspace(x0 - grid_extent, x0 + grid_extent, n)
        y = np.linspace(y0 - grid_extent, y0 + grid_extent, n)
        xx, yy = np.meshgrid(x, y)
        zz = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))
        zz = zz / zz.max()
        im = ax1.imshow(
            zz,
            extent=[
                x0 - grid_extent,
                x0 + grid_extent,
                y0 - grid_extent,
                y0 + grid_extent,
            ],
            origin="lower",
            cmap="inferno",
            alpha=0.6,
        )
        plt.colorbar(im, ax=ax1, label="Cloud density")

        plt.colorbar(pc, ax=ax1, label="Charge fraction")

        # Set limits
        ax1.set_xlim(params["x0_mm"] - 5, params["x0_mm"] + 5)
        ax1.set_ylim(params["y0_mm"] - 5, params["y0_mm"] + 5)

    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.set_title(f'Module {params["module"]}: {params["n_pads"]} pads hit')
    ax1.set_aspect("equal")

    # 2. Time distributions
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.bar(
        BIN_CENTERS,
        components["time_fractions"],
        width=P.adc_dt * 0.8,
        alpha=0.5,
        label="Before shaping",
    )
    ax2.plot(
        BIN_CENTERS,
        components["shaped_response"],
        "r-",
        linewidth=2,
        label="After shaping",
    )

    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Response (normalized)")
    ax2.set_title("Time Response")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Zoom to relevant range
    t_center = params["t_drift_ns"]
    t_range = 10 * max(params["sigma_total_t"], P.tau_shaper)
    ax2.set_xlim(max(0, t_center - t_range), t_center + t_range)

    # 3. Individual pad waveforms
    ax3 = fig.add_subplot(gs[0, 2])

    # Sort pads by charge
    sorted_pads = sorted(
        components["pad_fractions"].items(), key=lambda x: x[1], reverse=True
    )[:5]

    for pad_id, frac in sorted_pads:
        waveform = serf[(params["module"], pad_id)]
        ax3.plot(
            BIN_CENTERS,
            waveform * P.gain_mean * P.adc_conversion,
            label=f"Pad {pad_id}: {frac:.1%}",
        )

    ax3.set_xlabel("Time (ns)")
    ax3.set_ylabel("Signal (ADC units)")
    ax3.set_title("Expected Signals (mean gain)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(max(0, t_center - t_range), t_center + t_range)
    """
    # 4. Monte Carlo realization
    ax4 = fig.add_subplot(gs[1, :])

    # Generate MC realization
    wf = digitize_serf(serf, add_noise=True)

    # Plot waveforms
    for i, (pad_id, waveform) in enumerate(list(wf.items())[:5]):
        ax4.step(
            BIN_CENTERS,
            waveform + i * 20,
            where="mid",
            label=f"Pad {pad_id[1]}",
            linewidth=1.5,
        )

    ax4.axhline(
        P.adc_threshold, color="red", linestyle="--", label="Threshold", alpha=0.5
    )
    ax4.set_xlabel("Time (ns)")
    ax4.set_ylabel("ADC counts (offset for clarity)")
    ax4.set_title(f"MC Realization (Gain = {polya_gain():.0f})")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(max(0, t_center - t_range), t_center + t_range)
    """
    # 5. Statistics summary
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    stats_text = f"""
    ═══ Zigzag Pad SERF Summary ═══
    
    Position: ({x0:.1f}, {y0:.1f}, {z0:.1f}) cm
    Module: {params['module']} | Drift: {params['dz_cm']:.1f} cm in {params['t_drift_ns']:.0f} ns
    
    Cloud parameters:
    • Drift diffusion: σ_r = {params['sigma_drift_r']:.2f} mm, σ_t = {params['sigma_drift_t']:.1f} ns
    • GEM spread: σ_r = {P.sigma_aval_r:.2f} mm, σ_t = {P.sigma_aval_t:.1f} ns  
    • Total: σ_r = {params['sigma_total_r']:.2f} mm, σ_t = {params['sigma_total_t']:.1f} ns
    
    Response:
    • Pads hit: {params['n_pads']}
    • SAMPA shaping: τ = {P.tau_shaper:.0f} ns
    • Mean gain: {P.gain_mean:.0f} (Polya θ = {P.polya_theta:.1f})
    
    Electronics:
    • ADC: {P.adc_bits} bits @ {1000/P.adc_dt:.1f} MHz
    • Noise: {P.adc_noise:.1f} counts RMS
    • Threshold: {P.adc_threshold:.0f} counts
    """

    ax5.text(
        0.05,
        0.95,
        stats_text,
        transform=ax5.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(f"Zigzag Pad Response at ({x0}, {y0}, {z0}) cm", fontsize=16)
    return fig


def visualize_pad_plane_modules(df_centroids):
    """Visualize all three pad plane modules with fixed axis limits."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    module_names = ["Inner", "Middle", "Outer"]
    colors = ["red", "green", "blue"]

    # If tpc_x_sizes/tpc_y_sizes are in cm, convert to mm here:
    # mm_x = [[a*10, b*10] for (a, b) in tpc_x_sizes]
    # mm_y = [[a*10, b*10] for (a, b) in tpc_y_sizes]

    for module, (df, ax, name, color) in enumerate(
        zip(df_centroids, axes, module_names, colors)
    ):
        # Draw all pads
        patches = []
        for _, row in df.iterrows():
            verts = row["PadPath"].vertices
            patches.append(Polygon(verts, closed=True))
        pc = PatchCollection(
            patches, facecolors="none", edgecolors=color, linewidths=0.5
        )
        ax.add_collection(pc)

        # Use fixed limits
        x0, x1 = tpc_x_sizes[module]
        y0, y1 = tpc_y_sizes[module]
        # If working in mm, multiply by 10:
        # ax.set_xlim(x0*10, x1*10)
        # ax.set_ylim(y0*10, y1*10)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        ax.set_aspect("equal")
        ax.set_title(f"{name} Module ({len(df)} pads)")
        ax.set_xlabel("x (cm)")  # or 'x (mm)' if you scaled
        ax.set_ylabel("y (cm)")

        # Optional: draw a few reference circles at known radii
        r0, r1 = abs(x0), abs(x1)
        for rr in np.linspace(r0, r1, 5)[1:]:
            ax.add_patch(
                Circle((0, 0), rr, fill=False, color="gray", alpha=0.3, linestyle="--")
            )

    plt.suptitle("sPHENIX TPC Pad Plane Modules", fontsize=16)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("═══ Complete TPC Single Electron Response ═══\n")
    brd_maps = [
        "/Users/chenxi/developer/sphenix/tpc_sim/TPC_Toy_Model/PadPlane/AutoPad-R1-RevA.brd",
        "/Users/chenxi/developer/sphenix/tpc_sim/TPC_Toy_Model/PadPlane/AutoPad-R2-RevA-Pads.brd",
        "/Users/chenxi/developer/sphenix/tpc_sim/TPC_Toy_Model/PadPlane/AutoPad-R3-RevA.brd",
    ]
    sch_maps = [
        "/Users/chenxi/developer/sphenix/tpc_sim/TPC_Toy_Model/PadPlane/AutoPad-R1-RevA.sch",
        "/Users/chenxi/developer/sphenix/tpc_sim/TPC_Toy_Model/PadPlane/AutoPad-R2-RevA-Pads.sch",
        "/Users/chenxi/developer/sphenix/tpc_sim/TPC_Toy_Model/PadPlane/AutoPad-R3-RevA.sch",
    ]
    # Update paths as needed
    print("Loading pad planes...")
    print("(Update file paths in brd_maps if needed)\n")

    try:
        # Load pad planes
        df_centroids = load_pad_planes(brd_maps)

        # 1. Visualize pad modules
        print("1. Visualizing pad plane modules...")
        fig_modules = visualize_pad_plane_modules(df_centroids)

        # 2. Test positions in each module
        test_positions = [
            (35.0, 10.0, 50.0),  # Inner module
            (30.0, 20.0, 100.0),  # Middle module
            (45.0, 30.0, 75.0),  # Outer module
        ]

        for x0_cm, y0_cm, z0_cm in test_positions:
            print(f"\nTesting point ({x0_cm}, {y0_cm}, {z0_cm}) cm…")

            # convert to mm
            x0_mm = x0_cm * 10
            y0_mm = y0_cm * 10
            z0_mm = z0_cm * 10

            # rotate into sector 0
            x_rot, y_rot, sector = rotate_point_to_sector(x0_mm, y0_mm)
            print(f" → rotated into sector {sector}: ({x_rot:.1f}, {y_rot:.1f}) mm")

            module, layer = which_layer(np.hypot(x_rot, y_rot))
            print(f" → that lands in module {module}, layer {layer}")

            # mark it on the pad‐plane plot
            ax = fig_modules.axes[module]
            ax.scatter(
                [x_rot / 10.0],
                [y_rot / 10.0],
                marker="X",
                s=200,
                c="k",
                label=f"hit in sect ",
            )
            ax.legend(loc="upper right")
            plt.pause(0.1)

            serf, params, components = build_serf_zigzag_in_sector(
                x0_cm, y0_cm, z0_cm, df_centroids, return_components=True
            )

            # now you can plot it...
            fig = visualize_complete_response_zigzag(x0_cm, y0_cm, z0_cm, df_centroids)

            # …and gather some MC statistics
            n_mc = 20
            hits = []
            for _ in range(n_mc):
                wf = digitize_serf(serf, gain=polya_gain(), add_noise=True)
                hits.append(len(wf))
            print(
                f"   → pads above threshold: {np.mean(hits):.1f} ± {np.std(hits):.1f}"
            )

        plt.show()

    except FileNotFoundError as e:
        print(f"\nError: Could not find pad plane files.")
        print(f"Please update the paths in brd_maps to point to your AutoPad files.")
        print(f"Current paths:")
        for path in brd_maps:
            print(f"  - {path}")
        print(f"\nError details: {e}")

    print("\n═══ Analysis Complete ═══")
