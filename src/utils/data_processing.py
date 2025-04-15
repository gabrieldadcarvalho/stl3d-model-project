from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalize_mesh_data(vertices):
    scaler = MinMaxScaler()
    normalized_vertices = scaler.fit_transform(vertices)
    return normalized_vertices

def reshape_mesh_data(vertices, target_shape):
    reshaped_vertices = np.reshape(vertices, target_shape)
    return reshaped_vertices

def preprocess_mesh_data(mesh):
    vertices = mesh.vectors.reshape(-1, 3)  # Flatten the mesh vertices
    normalized_vertices = normalize_mesh_data(vertices)
    return normalized_vertices

def postprocess_output_data(output_data, original_shape):
    return output_data.reshape(original_shape)  # Reshape output to original mesh shape