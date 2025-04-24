import os
import pandas as pd
from rdkit import Chem
import torch
from COM_VAE import *
from Common import *
from Fragmentation import *
import warnings
from rdkit.Chem import Draw
from rdkit.Chem.QED import qed
from sascores import compute_sa_score
import subprocess
from collections import defaultdict
from grid import *
from rank import *
from pymol_visualization import *
from pdbqt import *
from run_docking import *
from PyPDF2 import PdfMerger
import argparse

warnings.filterwarnings("ignore", message=".*does not match num_features.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_checkpoint_path = "/home/chenyinghao/ll/dd/co_file/7/log/25Apr16_1123AM/epoch45_step26376.pt"
model = Space2Frag()
checkpoint = torch.load(model_checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
total_params = sum(p.numel() for p in model.parameters())
print(f"params:{total_params}")

def reconstruct_molecule_from_pdb(pdb_file_path, max_retries=5000):
    fragments = []
    for attempt in range(max_retries):
        try:
            pdb_batch = [pdb_file_path]
            S = model.sample(pdb_batch=pdb_batch).squeeze(0)
            end_index = (S == 2).nonzero(as_tuple=True)[0].item()
            S = S[:end_index]
            S = S.tolist()
            fragments = [vocab.get(idx) for idx in S if idx != 0]

            if len(fragments) < 2 and attempt < max_retries - 1:
                continue

            if len(fragments) >= 2:
                rdkit_mols = [mol_from_smiles(fragment) for fragment in fragments]
                reconstructed_mol, fragments_used = reconstruct(rdkit_mols)
                return reconstructed_mol, fragments
            
        except ValueError as e:
            break
        except Exception as e:
            break

    if len(fragments) < 2:
        if len(fragments) == 1:
            rdkit_mols = [mol_from_smiles(fragments[0])]
            reconstructed_mol, fragments_used = reconstruct(rdkit_mols)
            return reconstructed_mol, fragments
        else:
            return None, fragments

    return None, fragments

def save_molecule_to_sdf(smiles, output_file):
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            writer = Chem.SDWriter(output_file)
            writer.write(mol)
            writer.close()

def process_pdb_file_for_multiple_predictions(pdb_path, output_dir, target_count=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def get_current_sdf_count():
        return len([f for f in os.listdir(output_dir) if f.endswith('.sdf')])

    current_count = get_current_sdf_count()

    while current_count < target_count:
        reconstructed_mol, fragments = reconstruct_molecule_from_pdb(pdb_path)
        if reconstructed_mol:
            smiles = mol_to_smiles(reconstructed_mol)
            output_file = os.path.join(output_dir, f"molecule_{current_count + 1}.sdf")
            save_molecule_to_sdf(smiles, output_file)
            current_count += 1
        else:
            if len(fragments) == 1:
                smiles = mol_to_smiles(mol_from_smiles(fragments[0]))
                output_file = os.path.join(output_dir, f"molecule_{current_count + 1}.sdf")
                save_molecule_to_sdf(smiles, output_file)
                current_count += 1
            else:
                print(f"Skipping {pdb_path} due to reconstruction failure.")
        
        current_count = get_current_sdf_count()

def create_image_folder(parent_dir):
    image_dir = os.path.join(parent_dir, "image")
    os.makedirs(image_dir, exist_ok=True)  
    return image_dir

def create_pocket_folder(parent_dir):
    pocket_dir = os.path.join(parent_dir, "pocket")
    os.makedirs(pocket_dir, exist_ok=True)  
    return pocket_dir

def load_molecules_from_sdf(directory):
    molecules = []
    for filename in os.listdir(directory):
        if filename.endswith('.sdf'):
            sdf_path = os.path.join(directory, filename)
            try:
                supplier = Chem.SDMolSupplier(sdf_path)
                for mol in supplier:
                    if mol is not None:
                        molecules.append((mol, sdf_path))
                    else:
                        print(f"Warning: Skipping invalid molecule in {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return molecules

def process_molecules(molecules, output_img_dir):
    results = []
    os.makedirs(output_img_dir, exist_ok=True)  

    for idx, (mol, file_path) in enumerate(molecules, start=1):
        sa_score = compute_sa_score(mol)
        qed_score = qed(mol)
        total_score = sa_score + qed_score


        img_filename = f"molecule_{idx}.png"
        img_path = os.path.join(output_img_dir, img_filename)
        Draw.MolToFile(mol, img_path, size=(300, 300))

        results.append({
            "mole_id": idx,
            "mole_file": file_path,
            "mole_img": img_path,
            "score": total_score
        })
        results.sort(key=lambda x: x["score"], reverse=True)
        for new_id, res in enumerate(results, start=1):
            res["mole_id"] = new_id
  
    return results

def run_fpocket(pdb_file, output_dir="output"):
    cmd = ["fpocket", "-f", pdb_file, "-o", output_dir]
    subprocess.run(cmd, check=True)
    return output_dir

def get_pocket_output_path(pdb_path):
    pdb_dir, pdb_filename = os.path.split(pdb_path)  
    pdb_name, _ = os.path.splitext(pdb_filename)  
    pockets_dir = os.path.join(pdb_dir, f"{pdb_name}_out", "pockets")
    return pockets_dir

def extract_residue_numbers_from_folder(pdb_folder):
    chain_residues = defaultdict(set)  
    
    for pdb_file in os.listdir(pdb_folder):
        if pdb_file.endswith(".pdb"):  
            pdb_path = os.path.join(pdb_folder, pdb_file)
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        chain_id = line[21]
                        res_num = line[22:26].strip()
                        if res_num.isdigit():  
                            chain_residues[chain_id].add(int(res_num))

    for chain in chain_residues:
        chain_residues[chain] = sorted(chain_residues[chain])
    
    return chain_residues

def filter_pdb_by_residues(input_pdb, output_folder, chain_residues_dict):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(input_pdb, 'r') as f:
        output_files = {}
        
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]
                res_num = line[22:26].strip()
                
                if res_num.isdigit() and int(res_num) in chain_residues_dict.get(chain_id, []):
                    if chain_id not in output_files:
                        output_path = os.path.join(output_folder, f"Pocket_{chain_id}.pdb")
                        output_files[chain_id] = open(output_path, 'w')
                    
                    output_files[chain_id].write(line)
        
        for file in output_files.values():
            file.close()



def process_pdb_folder(pdb_folder, save_folder):
    results = []
    
    pdb_files = sorted([f for f in os.listdir(pdb_folder) if f.endswith('.pdb')])
    
    for pocket_id, pdb_file in enumerate(pdb_files, start=1):
        pdb_path = os.path.join(pdb_folder, pdb_file)
        
        pocket_save_folder = os.path.join(save_folder, f"pocket{pocket_id}")
        os.makedirs(pocket_save_folder, exist_ok=True)  
        
        process_pdb_file_for_multiple_predictions(pdb_path, pocket_save_folder)
        molecules = load_molecules_from_sdf(pocket_save_folder)
        image_folder = create_image_folder(pocket_save_folder)
        pocket_results = process_molecules(molecules, image_folder)
        
        results.append({
            'pocket_id': pocket_id,
            'pocket_file': pdb_path,
            'molecules': pocket_results
        })
    
    return results


def generate_results(pdb_path, save_folder, pocket_option='known'):
  if pocket_option=='known':
    process_pdb_file_for_multiple_predictions(pdb_path, save_folder) 
    molecules = load_molecules_from_sdf(save_folder)
    image_folder = create_image_folder(save_folder)
    results = process_molecules(molecules, image_folder)
  if pocket_option=='unknown':
    run_fpocket(pdb_path)
    fpocket_out = get_pocket_output_path(pdb_path)
    chain_residues_dict = extract_residue_numbers_from_folder(fpocket_out)
    pre_pocket_folder = os.path.join(save_folder, "pre_pocket")
    os.makedirs(pre_pocket_folder, exist_ok=True)
    filter_pdb_by_residues(pdb_path, pre_pocket_folder, chain_residues_dict)
    results = process_pdb_folder(pre_pocket_folder, save_folder)
    
  results_path = os.path.join(save_folder, "results.txt")
  with open(results_path, 'w') as f:
      if isinstance(results, (list, tuple)):
          for item in results:
              f.write(f"{item}\n")
      elif isinstance(results, dict):
          for key, value in results.items():
              f.write(f"{key}: {value}\n")
      else:
          f.write(str(results))
  return results
    
def generate_combined_report(input_dir, top_n=2):
    """Generate combined PDF report for all pocket_* subdirectories
    
    Args:
        input_dir (str): Directory containing pocket_* subfolders
        top_n (int): Number of top molecules to display per pocket (default: 2)
    """
    # Validate input directory
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_pdf = os.path.join(input_dir, "combined_report.pdf")
    
    # Find all pocket subdirectories
    pocket_dirs = sorted([
        d for d in os.listdir(input_dir) 
        if (os.path.isdir(os.path.join(input_dir, d)) and 
            d.lower().startswith("pocket_"))
    ])
    
    if not pocket_dirs:
        raise FileNotFoundError(f"No pocket_* subdirectories found in {input_dir}")
    
    print(f"Found {len(pocket_dirs)} pocket directories:")
    for idx, pocket in enumerate(pocket_dirs, 1):
        print(f"{idx}. {pocket}")

    merger = PdfMerger()
    temp_files = []
    
    try:
        # Process each pocket
        for pocket in pocket_dirs:
            pocket_path = os.path.join(input_dir, pocket)
            temp_pdf = os.path.join(input_dir, f"TEMP_{pocket}.pdf")
            
            print(f"\nProcessing: {pocket}...")
            run_visualization(
                input_dir=pocket_path,
                output_file=temp_pdf,
                top_n=top_n
            )
            
            merger.append(temp_pdf)
            temp_files.append(temp_pdf)
            print(f"Added: {pocket} to combined report")
        
        # Generate final output
        merger.write(output_pdf)
        print(f"\nSuccessfully generated combined report: {output_pdf}")
        print(f"Total pages merged: {len(merger.pages)}")
    
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
    finally:
        # Cleanup resources
        merger.close()
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                print(f"Warning: Could not delete temp file {f}: {str(e)}")


#pdb_path="/home/chenyinghao/ll/dd/co_file/7/fpocket/6qep/chain_A.pdb"
#save_folder="/home/chenyinghao/ll/dd/co_file/7/generated_molecules11/"
#results = generate_results(pdb_path, save_folder, pocket_option='known')
#print(results)

#pdb_path="/home/chenyinghao/ll/dd/co_file/7/6cqe.pdb"
#save_folder="/home/chenyinghao/ll/dd/co_file/7/generated_molecules17/"
#results = generate_results(pdb_path, save_folder, pocket_option='unknown')
#print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="a")
    parser.add_argument("pdb_path", type=str, help="b")
    parser.add_argument("save_folder", type=str, help="c")
    parser.add_argument("pocket_option", type=str, default="unknown", help="d")
    args = parser.parse_args()
    
    generate_results(args.pdb_path, args.save_folder, args.pocket_option)
    convert_sdf_to_pdbqt(args.save_folder)
    convert_pdb_to_pdbqt(args.pdb_path)
    calculate_docking_parameters(args.save_folder, edge=5.0)
    results = run_multi_pocket_docking(args.pdb_path, args.save_folder)
    batch_process(args.save_folder,args.pdb_path)
    generate_combined_report(args.save_folder, 2)

    