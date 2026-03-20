import os
import shutil
import random
from collections import defaultdict
import numpy as np

def get_pose_category(filename):
    if not filename.endswith('.png'):
        return None
    
    if filename.startswith('exp_') or filename.startswith('gen_'):
        filename = filename[4:]
    
    parts = filename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return None

def analyze_data_distribution(data_dir):
    category_files = defaultdict(list)
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            category = get_pose_category(filename)
            if category:
                category_files[category].append(filename)
    
    return category_files

def create_hybrid_dataset(exp_dir, gen_dir, output_dir, exp_ratio, gen_ratio):
    print(f"\n Hybrid Data: {exp_ratio}% Exp + {gen_ratio}% Gen")
    print(f"Output: {output_dir}")
    
    exp_categories = analyze_data_distribution(exp_dir)
    gen_categories = analyze_data_distribution(gen_dir)
    
    all_categories = set(exp_categories.keys()) | set(gen_categories.keys())
    print(f"All Class Num: {len(all_categories)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_copied = 0
    
    for category in sorted(all_categories):
        exp_files = exp_categories.get(category, [])
        gen_files = gen_categories.get(category, [])
        
        print(f"Class {category}: Exp={len(exp_files)}, Gen={len(gen_files)}")

        exp_count = int(len(exp_files) * exp_ratio / 100) if exp_files else 0
        gen_count = int(len(gen_files) * gen_ratio / 100) if gen_files else 0
        
        print(f"  -> Choose Exp={exp_count}, Gen={gen_count}")
        
        selected_exp = random.sample(exp_files, exp_count) if exp_count > 0 else []
        selected_gen = random.sample(gen_files, gen_count) if gen_count > 0 else []
        

        for filename in selected_exp:
            src_path = os.path.join(exp_dir, filename)
            new_filename = f"exp_{filename}"
            dst_path = os.path.join(output_dir, new_filename)
            shutil.copy2(src_path, dst_path)
            total_copied += 1
            
        for filename in selected_gen:
            src_path = os.path.join(gen_dir, filename)
            new_filename = f"gen_{filename}"
            dst_path = os.path.join(output_dir, new_filename)
            shutil.copy2(src_path, dst_path)
            total_copied += 1
    
    print(f"Copy {total_copied} Files")
    return total_copied

def analyze_and_print_distribution(data_dir, dataset_name):
    print(f"\n=== {dataset_name} Analysis ===")
    category_files = analyze_data_distribution(data_dir)
    
    total_files = sum(len(files) for files in category_files.values())
    print(f"All Files Num: {total_files}")
    print(f"Class Num: {len(category_files)}")
    
    exp_count = 0
    gen_count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            if filename.startswith('exp_'):
                exp_count += 1
            elif filename.startswith('gen_'):
                gen_count += 1
    
    print(f"Exp File Num: {exp_count}")
    print(f"Gen File Num: {gen_count}")
    
    for category in sorted(category_files.keys()):
        count = len(category_files[category])
        print(f"  {category}: {count} 个文件")
    
    return category_files

def main():
    base_dir = "./"
    exp_dir = os.path.join(base_dir, "Experiment_All_separate/Train")
    gen_dir = os.path.join(base_dir, "Generated_All_separate/Train")
    
    if not os.path.exists(exp_dir):
        print(f"Wrong: {exp_dir}")
        return
    if not os.path.exists(gen_dir):
        print(f"Wrong: {gen_dir}")
        return
    
    exp_dist = analyze_and_print_distribution(exp_dir, "Experiment_All")
    gen_dist = analyze_and_print_distribution(gen_dir, "Generated_All")
    
    random.seed(42)
    np.random.seed(42)
    
    hybrid_configs = [
        (75, 25, "Hybrid_75Exp_25Gen"),
        (50, 50, "Hybrid_50Exp_50Gen"), 
        (25, 75, "Hybrid_25Exp_75Gen")
    ]
    
    for exp_ratio, gen_ratio, output_name in hybrid_configs:
        output_dir = os.path.join(base_dir, f"{output_name}_separate/Train")
        create_hybrid_dataset(exp_dir, gen_dir, output_dir, exp_ratio, gen_ratio)
    
    print("\n All Done！")
    
    print("\n=== Val Result ===")
    for exp_ratio, gen_ratio, output_name in hybrid_configs:
        output_dir = os.path.join(base_dir, f"{output_name}_separate/Train")
        if os.path.exists(output_dir):
            file_count = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
            print(f"{output_name}: {file_count} Files")
            analyze_and_print_distribution(output_dir, output_name)

if __name__ == "__main__":
    main() 