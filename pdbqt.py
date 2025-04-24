import os
import subprocess
import sys

def convert_sdf_to_pdbqt(save_folder):
    for folder_name in os.listdir(save_folder):
        folder_path = os.path.join(save_folder, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".sdf"):
                sdf_path = os.path.join(folder_path, file_name)
                pdbqt_path = os.path.splitext(sdf_path)[0] + ".pdbqt"
                
                cmd = [
                    "obabel",
                    "-isdf", sdf_path,
                    "-opdbqt",
                    "-O", pdbqt_path
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    print(f"Conversion successful: {sdf_path} -> {pdbqt_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Conversion failed: {sdf_path}")
                    print(f"Error message: {e}")
                except Exception as e:
                    print(f"Unexpected error: {e}")

def convert_pdb_to_pdbqt(pdb_path):
    if not os.path.isfile(pdb_path):
        print(f"Error: Input file {pdb_path} not found")
        return False
    
    if not pdb_path.lower().endswith('.pdb'):
        print(f"Error: {pdb_path} is not a PDB file")
        return False

    pdbqt_path = os.path.splitext(pdb_path)[0] + '.pdbqt'
    
    try:
        subprocess.run(
            ['prepare_receptor', '-r', pdb_path, '-o', pdbqt_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print(f"Conversion successful: {pdb_path} -> {pdbqt_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {pdb_path}")
        print(f"Error details: {e.stderr.decode().strip()}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False


#convert_sdf_to_pdbqt("/home/chenyinghao/ll/dd/co_file/7/generated_molecules9/")
#convert_pdb_to_pdbqt("/home/chenyinghao/ll/dd/co_file/7/6cqe.pdb")
#
#if __name__ == "__main__":
#    if len(sys.argv) != 2:
#        print("Usage: python converter.py <save_folder>")
#        sys.exit(1)
#    
#    target_folder = sys.argv[1]
#    
#    if not os.path.exists(target_folder):
#        print(f"Error: Directory {target_folder} does not exist")
#        sys.exit(1)
#    
#    convert_sdf_to_pdbqt(target_folder)
#    print("All files converted successfully!")

import os
import subprocess
import sys

def convert_sdf_to_pdbqt(ligand_folder):
    """Convert all SDF files in ligand folder (including subfolders) to PDBQT with '_m' suffix"""
    for root, dirs, files in os.walk(ligand_folder):
        for file_name in files:
            if file_name.lower().endswith(".sdf"):
                sdf_path = os.path.join(root, file_name)
                
                base_name = os.path.splitext(file_name)[0]
                pdbqt_name = f"{base_name}.pdbqt"
                pdbqt_path = os.path.join(root, pdbqt_name)
                
                cmd = [
                    "obabel",
                    "-isdf", sdf_path,
                    "-opdbqt",
                    "-O", pdbqt_path
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    print(f"Conversion successful: {sdf_path} -> {pdbqt_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Conversion failed: {sdf_path}")
                    print(f"Error message: {e}")
                except Exception as e:
                    print(f"Unexpected error: {e}")

def convert_pdb_to_pdbqt(pdb_path):
    """Convert single PDB file to PDBQT using prepare_receptor"""
    if not os.path.isfile(pdb_path):
        print(f"Error: Input file {pdb_path} not found")
        return False
    
    if not pdb_path.lower().endswith('.pdb'):
        print(f"Error: {pdb_path} is not a PDB file")
        return False

    pdbqt_path = os.path.splitext(pdb_path)[0] + '.pdbqt'
    
    try:
        subprocess.run(
            ['prepare_receptor', '-r', pdb_path, '-o', pdbqt_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print(f"Conversion successful: {pdb_path} -> {pdbqt_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {pdb_path}")
        print(f"Error details: {e.stderr.decode().strip()}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def convert_pdb_folder_to_pdbqt(protein_folder):
    for root, dirs, files in os.walk(protein_folder):
        for file_name in files:
            if file_name.lower().endswith('.pdb'):
                pdb_path = os.path.join(root, file_name)
                convert_pdb_to_pdbqt(pdb_path)

#if __name__ == "__main__":
#    PROTEIN_FOLDER = "/home/chenyinghao/ll/dd/co_file/7/ligand/"
#    LIGAND_FOLDER = "/home/chenyinghao/ll/dd/co_file/7/predict_attn_1/"
#    
#    print("Processing protein files...")
#    convert_pdb_folder_to_pdbqt(PROTEIN_FOLDER)
#    
#    print("\nProcessing ligand files...")
#    convert_sdf_to_pdbqt(LIGAND_FOLDER)
#    
#    print("\nAll conversions completed!")