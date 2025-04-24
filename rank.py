import os
import glob
import re
import shutil
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from sascores import compute_sa_score
from copy import deepcopy

def get_smiles(mol):
    try:
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    except Exception as e:
        print(f"Error generating SMILES: {str(e)}")
        return "N/A"
        
def get_logp(mol):
    return Crippen.MolLogP(mol)

def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return sum([rule_1, rule_2, rule_3, rule_4, rule_5])

def get_tpsa(mol):
    return Descriptors.TPSA(mol)

def process_pocket_results(result_folder, global_file_path):
    folder_name = os.path.basename(result_folder)
    match = re.match(r'pocket(\d+)_results$', folder_name)
    if not match:
        print(f"Skipping invalid folder: {folder_name}")
        return
    
    pocket_num = match.group(1)
    parent_dir = os.path.dirname(result_folder)
    
    try:
        pocket_num_int = int(pocket_num)
        if pocket_num_int < 1 or pocket_num_int > 26:
            raise ValueError
    except ValueError:
        print(f"Invalid pocket number: {pocket_num}")
        return
    
    source_folder = os.path.join(parent_dir, f"pocket{pocket_num}")
    image_folder = os.path.join(source_folder, "image")
    
    if not os.path.exists(source_folder):
        print(f"Missing source folder: {source_folder}")
        return
    if not os.path.exists(image_folder):
        print(f"Missing image folder: {image_folder}")
        return

    log_files = glob.glob(os.path.join(result_folder, 'molecule_*_log.txt'))
    score_records = []

    for log_file in log_files:
        mol_id_match = re.search(r'molecule_(\d+)_log', log_file)
        if not mol_id_match:
            continue
        mol_id = mol_id_match.group(1)
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {log_file}: {str(e)}")
            continue

        best_score = None
        for i, line in enumerate(lines):
            if 'mode' in line and 'affinity' in line:
                for j in range(i+1, min(i+5, len(lines))):
                    if re.match(r'^[-+]+', lines[j].strip()):
                        if j+1 < len(lines):
                            data_line = lines[j+1].strip().split()
                            if len(data_line) >= 2:
                                best_score = data_line[1]
                        break
                break

        if best_score:
            score_filename = f"molecule_{mol_id}.txt"
            score_file = os.path.join(result_folder, score_filename)
            
            sdf_file = os.path.join(source_folder, f"molecule_{mol_id}.sdf")
            
            try:
                suppl = Chem.SDMolSupplier(sdf_file)
                mol = next(suppl, None)
                if mol:
                    smiles = get_smiles(mol)
                    logp = get_logp(mol)
                    lipinski = obey_lipinski(mol)
                    sa_score = compute_sa_score(mol)
                    qed_score = QED.qed(mol)
                    tpsa = get_tpsa(mol)
                    
                    with open(score_file, 'w') as f:
                        f.write(f"SMILES: {smiles}\n")
                        f.write(f"vina score: {best_score}\n")
                        f.write(f"LogP: {logp:.2f}\n")
                        f.write(f"Lipinski Compliance: {lipinski}/5\n")
                        f.write(f"Synthetic Accessibility: {sa_score:.2f}\n")
                        f.write(f"QED: {qed_score:.2f}\n")
                        f.write(f"TPSA: {tpsa:.2f}\n")
                else:
                    print(f"Invalid molecule in {sdf_file}")
                    continue
            except Exception as e:
                print(f"Error processing {sdf_file}: {str(e)}")
                continue

            pdbqt_file = log_file.replace('_log.txt', '_complex.pdbqt')
            png_file = os.path.join(image_folder, f"molecule_{mol_id}.png")
            
            if all(os.path.exists(f) for f in [pdbqt_file, sdf_file, png_file]):
                score_records.append({
                    'score': float(best_score),
                    'mol_id': mol_id,
                    'files': {
                        'score': score_file,
                        'pdbqt': pdbqt_file,
                        'sdf': sdf_file,
                        'png': png_file
                    }
                })
            else:
                print(f"Missing files for molecule {mol_id}")

    sorted_scores = sorted(score_records, key=lambda x: x['score'])[:2]
    if not sorted_scores:
        print(f"No valid molecules found in {folder_name}")
        return  
    pocket_letter = chr(64 + int(pocket_num))
    target_folder = os.path.join(parent_dir, f"pocket_{pocket_letter}")
    os.makedirs(target_folder, exist_ok=True)

    for entry in sorted_scores:
        for ftype in ['score', 'pdbqt', 'sdf']:
            src = entry['files'][ftype]
            dst = os.path.join(target_folder, os.path.basename(src))
            shutil.copy(src, dst)

        png_src = entry['files']['png']
        png_dst = os.path.join(target_folder, os.path.basename(png_src))
        shutil.copy(png_src, png_dst)

    pre_pocket_dir = os.path.join(parent_dir, 'pre_pocket')
    pdb_src = os.path.join(pre_pocket_dir, f"Pocket_{pocket_letter}.pdb")
    if os.path.exists(pdb_src):
        shutil.copy(pdb_src, os.path.join(target_folder, f"Pocket_{pocket_letter}.pdb"))
        
    if global_file_path and os.path.isfile(global_file_path):
        try:
            shutil.copy(global_file_path, target_folder)
            print(f"Copied global file to {os.path.basename(target_folder)}")
        except Exception as e:
            print(f"Failed to copy global file: {str(e)}")
    elif global_file_path:
        print(f"Global file not found: {global_file_path}")

    print(f"Processed {len(sorted_scores)} molecules in {folder_name}")



def batch_process(root_directory, global_file_path=None):
    result_folders = glob.glob(os.path.join(root_directory, 'pocket*_results'))
    
    for result_folder in result_folders:
        if os.path.isdir(result_folder):
            print(f"\nProcessing {os.path.basename(result_folder)}...")
            process_pocket_results(result_folder, global_file_path)

    print("\nBatch processing completed!")


#batch_process("/home/chenyinghao/ll/dd/co_file/7/generated_molecules19/","/home/chenyinghao/ll/dd/co_file/7/6cqe.pdb")