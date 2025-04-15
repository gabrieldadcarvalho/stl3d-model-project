from models.stl3d_model import STL3DModel
from utils.stl_io import load_stl_data, save_stl_data
import torch

def main():
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STL3DModel().to(device)

    # Load input STL file
    input_stl_path = 'path/to/input_file.stl'  # Replace with actual input file path
    input_data = load_stl_data(input_stl_path)

    # Process the input data
    output_data = model(input_data.to(device))

    # Save the output STL file
    output_stl_path = 'path/to/output_file.stl'  # Replace with actual output file path
    save_stl_data(output_stl_path, output_data.detach().cpu().numpy())

if __name__ == "__main__":
    main()