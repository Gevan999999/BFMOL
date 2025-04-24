import os
import time
import subprocess
import glob

def create_complex(receptor_path, ligand_path, output_path):
    try:
        with open(receptor_path, 'r') as rec, open(ligand_path, 'r') as lig:
            with open(output_path, 'w') as f:
                f.write(rec.read() + "\n" + lig.read())
        return True
    except Exception as e:
        print(f"Complex creation failed: {str(e)}")
        return False

def run_vina_with_timeout(cmd, timeout_sec=300):
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
            return proc.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            proc.kill()
            return -1, "", f"Timeout after {timeout_sec}s"
    except Exception as e:
        return -2, "", f"Execution error: {str(e)}"

def process_pocket(receptor, conf_file, ligand_folder, output_root, timeout_sec=300):
    start_time = time.time()
    pocket_name = os.path.basename(ligand_folder)
    output_dir = os.path.join(output_root, f"{pocket_name}_results")
    os.makedirs(output_dir, exist_ok=True)

    ligand_files = [f for f in os.listdir(ligand_folder) if f.endswith('.pdbqt')]
    results = []
    
    print(f"\nProcessing pocket: {pocket_name} ({len(ligand_files)} molecules)")
    
    for lig_file in ligand_files:
        lig_path = os.path.join(ligand_folder, lig_file)
        base_name = os.path.splitext(lig_file)[0]
        output_file = os.path.join(output_dir, f"{base_name}_out.pdbqt")
        log_file = os.path.join(output_dir, f"{base_name}_log.txt")
        complex_file = os.path.join(output_dir, f"{base_name}_complex.pdbqt")

        result = {
            'pocket': pocket_name,
            'ligand': lig_file,
            'output': output_file,
            'complex': complex_file,
            'score': None,
            'error': None,
            'time_sec': 0
        }

        mol_start = time.time()
        cmd = [
            'vina',
            '--receptor', receptor,
            '--config', conf_file,
            '--ligand', lig_path,
            '--out', output_file,
            '--log', log_file
        ]
        
        retcode, stdout, stderr = run_vina_with_timeout(cmd, timeout_sec=timeout_sec)
        
        if retcode == 0:
            create_complex(receptor, output_file, complex_file)
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if line.strip() and line[0].isdigit():
                            parts = line.split()
                            if len(parts) >= 2:
                                result['score'] = float(parts[1])
                                break
            except Exception as e:
                print(f"  Error reading scores: {str(e)}")
        elif retcode == -1:
            result['error'] = f"Timeout after {timeout_sec}s"
        elif retcode == -2:
            result['error'] = stderr
        else:
            result['error'] = f"Docking failed: {stderr.strip()}"
        
        result['time_sec'] = time.time() - mol_start
        results.append(result)
        
        score_display = f"{result['score']:6.2f}" if result['score'] is not None else "  N/A  "
        if result['error']:
            print(f"  {lig_file[:15]:<15} | Failed: {result['error'][:50]}")
        else:
            print(f"  {lig_file[:15]:<15} | Success | Score: {score_display} | Time: {result['time_sec']:5.1f}s")

    total_time = time.time() - start_time
    print(f"Finished {pocket_name} | Total time: {total_time/60:.1f} mins\n")
    return results

def generate_pdbqt_path(pdb_path):
    base_path = os.path.splitext(pdb_path)[0]  
    return base_path + '.pdbqt'

def run_multi_pocket_docking(pdb_path, save_folder, timeout_sec=300):
    receptor = generate_pdbqt_path(pdb_path)
    total_start = time.time()
    conf_dir = os.path.join(save_folder, "pre_pocket")
    conf_files = sorted(glob.glob(os.path.join(conf_dir, "*_conf.txt")))
    
    all_results = {}
    
    print(f"\n{'='*50}\nStarting multi-pocket docking with {len(conf_files)} pockets\n{'='*50}")
    
    for conf_path in conf_files:
        conf_name = os.path.basename(conf_path)
        pocket_id = conf_name.split("_")[1].lower()
        pocket_num = ord(pocket_id) - ord('a') + 1
        ligand_dir = os.path.join(save_folder, f"pocket{pocket_num}")
        
        if not os.path.exists(ligand_dir):
            print(f"Skipping {conf_name}: Corresponding ligand folder not found")
            continue
            
        results = process_pocket(
            receptor=receptor,
            conf_file=conf_path,
            ligand_folder=ligand_dir,
            output_root=save_folder,
            timeout_sec=timeout_sec
        )
        
        all_results[f"pocket{pocket_num}"] = results

    total_time = time.time() - total_start
    print(f"\n{'='*50}\nTotal docking completed | Pockets: {len(conf_files)} | Total time: {total_time/60:.1f} mins\n{'='*50}")
    return all_results


#results = run_multi_pocket_docking(
#   pdb_path="/home/chenyinghao/ll/dd/co_file/7/6cqe.pdb",
#   save_folder="/home/chenyinghao/ll/dd/co_file/7/generated_molecules19/",
#   timeout_sec=20  
#)
