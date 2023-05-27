import torch
import numpy as np

def generate_azel_xyz_grid(n_az, n_el, az_lim=(-np.pi, np.pi),
                           el_lim=(-np.pi/2, np.pi/2)):
    """Generate a grid of points in azimuth and elevation, and convert to
    Cartesian coordinates.
    """
    az = np.linspace(*az_lim, n_az)
    el = np.linspace(*el_lim, n_el)
    az, el = np.meshgrid(az, el)
    az = az.flatten()
    el = el.flatten()
    x, y, z = np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)
    return np.stack([x, y, z], axis=1)


def generate_orthogonal_vectors_np(v1):
    """Generate two random orthogonal vectors to v1.
    """
    # Ensure v1 is a torch tensor
    if not isinstance(v1, np.ndarray):
        v1 = np.array(v1)
    # Generate two random vectors
    v2 = np.random.randn(*v1.shape)
    v3 = np.random.randn(*v1.shape)
    # Make v2 orthogonal to v1
    v2 -= (v2 @ v1) / np.linalg.norm(v1)**2 * v1
    # Make v3 orthogonal to both v1 and v2
    v3 -= (v3 @ v1) / np.linalg.norm(v1)**2 * v1
    v3 -= (v3 @ v2) / np.linalg.norm(v2)**2 * v2
    # Normalize v2 and v3 to have the same length as v1
    norm_v1 = np.linalg.norm(v1)
    v2 = v2 * norm_v1 / np.linalg.norm(v2)
    v3 = v3 * norm_v1 / np.linalg.norm(v3)
    return v2, v3


def generate_orthogonal_vectors_torch(v1):
    """Generate two random orthogonal vectors to v1.
    """
    # Ensure v1 is a torch tensor
    if not isinstance(v1, torch.Tensor):
        v1 = torch.tensor(v1)
    # Generate two random vectors
    v2 = torch.randn(v1.shape)
    v3 = torch.randn(v1.shape)
    # Make v2 orthogonal to v1
    v2 -= (v2 @ v1) / v1.norm().pow(2) * v1
    # Make v3 orthogonal to both v1 and v2
    v3 -= (v3 @ v1) / v1.norm().pow(2) * v1
    v3 -= (v3 @ v2) / v2.norm().pow(2) * v2
    # Normalize v2 and v3 to have the same length as v1
    norm_v1 = v1.norm()
    v2 = v2 * norm_v1 / v2.norm()
    v3 = v3 * norm_v1 / v3.norm()
    return v2, v3


def generate_sphere_grid_coords(vec1, vec2=None, vec3=None,
                                n_az=9, n_el=9,
                          az_lim=(-np.pi/2, np.pi/2),
                          el_lim=(-np.pi/2, np.pi/2)):
    coords = generate_azel_xyz_grid(n_az, n_el, az_lim=az_lim,
                                    el_lim=el_lim)
    if not isinstance(vec1, torch.Tensor):
        vec1 = torch.tensor(vec1)
    if vec2 is None or vec3 is None:
        # torch.random.manual_seed(0)
        vec2, vec3 = generate_orthogonal_vectors_torch(vec1)
    basis = torch.stack([vec1, vec2, vec3], dim=0)
    codes = torch.matmul(torch.tensor(coords, dtype=torch.float32), basis)
    return codes
