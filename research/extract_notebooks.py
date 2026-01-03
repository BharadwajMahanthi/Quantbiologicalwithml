import json
import os

def extract_notebook(filepath, output_path):
    print(f"Extracting {filepath} to {output_path}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Extracted content from {os.path.basename(filepath)}\n\n")
        
        cells = nb.get('cells', [])
        for i, cell in enumerate(cells):
            cell_type = cell.get('cell_type')
            source = cell.get('source', [])
            if isinstance(source, str):
                 source = [source]
            
            content = "".join(source)
            
            if not content.strip():
                continue

            if cell_type == 'markdown':
                f.write(f"## Cell {i+1} (Markdown)\n\n")
                f.write(content)
                f.write("\n\n")
            elif cell_type == 'code':
                f.write(f"## Cell {i+1} (Code)\n\n")
                f.write("```python\n")
                f.write(content)
                f.write("\n```\n\n")
    
    print(f"Done extracting {filepath}")

def main():
    base_dir = r"C:\Users\mbpd1\Downloads\Quantbiologicalwithml"
    files = [
        ("life_equ v.2.ipynb", "life_equ_v.2_extracted.md"),
        ("life_equ.ipynb", "life_equ_extracted.md")
    ]

    for nb_file, out_file in files:
        nb_path = os.path.join(base_dir, nb_file)
        out_path = os.path.join(base_dir, out_file)
        if os.path.exists(nb_path):
            extract_notebook(nb_path, out_path)
        else:
            print(f"File not found: {nb_path}")

if __name__ == "__main__":
    main()
