from stl import mesh
import numpy as np

def load_stl(file_path):
    """Load an STL file and return the mesh data."""
    return mesh.Mesh.from_file(file_path)

def save_stl(file_path, mesh_data):
    """Save mesh data to an STL file."""
    mesh_data.save(file_path)

def mesh_to_array(mesh_data):
    """Convert mesh data to a NumPy array."""
    return np.array([mesh_data.v0, mesh_data.v1, mesh_data.v2])

def array_to_mesh(array_data):
    """Convert a NumPy array back to mesh data."""
    return mesh.Mesh(np.zeros(array_data.shape[0], dtype=mesh.Mesh.dtype)).from_array(array_data)