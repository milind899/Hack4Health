import os
import re

def remove_comments(text):
    # Remove single line comments
    text = re.sub(r'#.*', '', text)
    # Remove multiline comments (docstrings)
    text = re.sub(r'"""[\s\S]*?"""', '', text)
    text = re.sub(r"'''[\s\S]*?'''", '', text)
    # Remove extra empty lines created by comment removal
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def aggregate_code():
    print(">>> 📜 Aggregating source code and removing comments...")
    source_files = [
        "config.py",
        "predict.py",
        "scripts/1_data_loading.py",
        "scripts/2_preprocessing.py",
        "scripts/3_baseline_model.py",
        "scripts/4_semantic_model.py",
        "scripts/4b_mlp_classifier.py",
        "scripts/5_transformer_model.py",
        "scripts/6_ensemble_model.py",
        "scripts/7_8_error_analysis.py",
        "scripts/9_performance_report.py"
    ]
    
    with open("COMPLETE_SOURCE_CODE.py", "w", encoding="utf-8") as outfile:
        outfile.write('"""\nCOMPLETE SOURCE CODE FOR HACK4HEALTH\n' + "="*40 + '\n"""\n\n')
        
        for file_path in source_files:
            if os.path.exists(file_path):
                outfile.write(f"\n\n# {'='*40}\n# FILE: {file_path}\n# {'='*40}\n\n")
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    clean_content = remove_comments(content)
                    outfile.write(clean_content)
                    outfile.write("\n")
            else:
                print(f"Warning: File {file_path} not found.")

    print("✅ SUCCESS: All code aggregated in COMPLETE_SOURCE_CODE.py (Comment-free)")

if __name__ == "__main__":
    aggregate_code()
