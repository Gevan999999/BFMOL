import os
import numpy as np

def calculate_docking_parameters(folder_path, edge=5.0):
    pre_pocket_dir = os.path.join(folder_path, "pre_pocket")
    
    if not os.path.exists(pre_pocket_dir):
        raise FileNotFoundError(f"pre_pocket directory not found in {folder_path}")
    
    pdb_files = [f for f in os.listdir(pre_pocket_dir) if f.endswith(".pdb")]
    
    for pdb_file in pdb_files:
        pdb_path = os.path.join(pre_pocket_dir, pdb_file)
        output_conf = os.path.join(pre_pocket_dir, f"{os.path.splitext(pdb_file)[0]}_conf.txt")
        
        coords = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])
                    except:
                        continue
        
        if not coords:
            print(f"Skipped {pdb_file}: No valid coordinates found")
            continue
        
        coords = np.array(coords)
        center = np.mean(coords, axis=0).round(3)
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        size = (max_coords - min_coords + edge).round(3)
        
        config_content = f"""receptor = receptor.pdbqt
center_x = {center[0]}
center_y = {center[1]}
center_z = {center[2]}
size_x = {size[0]}
size_y = {size[1]}
size_z = {size[2]}
num_modes = 10
exhaustiveness = 8
"""
        with open(output_conf, 'w') as f:
            f.write(config_content)
        print(f"Generated: {output_conf}")

#if __name__ == "__main__":
#    import sys
#    if len(sys.argv) != 2:
#        print("Usage: python script.py <folder_path>")
#        sys.exit(1)
#    
#    try:
#        calculate_docking_parameters(sys.argv[1])
#        print("Processing completed")
#    except Exception as e:
#        print(f"Error: {str(e)}")
#        sys.exit(1)

#calculate_docking_parameters("/home/chenyinghao/ll/dd/co_file/7/generated_molecules19/", edge=5.0)