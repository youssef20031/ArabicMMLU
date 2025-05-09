import subprocess
import sys
import os
import ast # For parsing Python code to find imports
import re

REQUIREMENTS_IN_FILE = "requirements.in"
REQUIREMENTS_TXT_FILE = "requirements.txt"

# Common direct dependencies (example, will be populated if requirements.in is created)
SUGGESTED_DIRECT_DEPENDENCIES = [
    "pandas", "peft", "transformers", "torch", "torchvision", "torchaudio",
    "tqdm", "groq", "httpx", "accelerate", "bitsandbytes", "sentencepiece", "safetensors"
]

# --- New functions for scanning project dependencies ---

def get_stdlib_modules():
    """Returns a set of Python standard library module names."""
    if sys.version_info >= (3, 10):
        return sys.stdlib_module_names
    else:
        # A fallback list for older Python versions (might not be exhaustive)
        # This is less critical if the user's environment is Python 3.10+
        print("Warning: Using a fallback list of stdlib modules for Python < 3.10. Consider upgrading Python for more accurate stdlib detection.")
        # This list can be quite long. For brevity, I'll keep it short.
        # A more robust solution for <3.10 would be to install `stdlibs` package.
        return {
            'os', 'sys', 'math', 'json', 'argparse', 'datetime', 'collections',
            'itertools', 'functools', 're', 'logging', 'threading', 'subprocess',
            'time', 'ast', 'importlib'
            # Add more common ones if needed, or use a more comprehensive list
        }

def find_python_files(project_root="."):
    """Finds all .py files in the project, excluding common venv directories."""
    py_files = []
    excluded_dirs = {'.venv', 'venv', 'env', '__pycache__', '.git', '.vscode'}
    for root, dirs, files in os.walk(project_root):
        # Modify dirs in-place to prevent walk from descending into excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

def extract_imports_from_file(file_path):
    """Parses a Python file and extracts top-level imported module names."""
    imports = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.level == 0: # Absolute import
                        imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"Warning: Could not parse {file_path} for imports: {e}")
    return imports

def get_dependencies_from_in_file(filepath):
    """Reads requirements.in and extracts package names."""
    deps = set()
    if not os.path.exists(filepath):
        return deps
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Use re.split to split by any of the version specifiers or space
                # The pattern '[=<>!~ ]+' will split by one or more occurrences of these characters
                dep_name = re.split(r'[=<>!~ ]+', line, maxsplit=1)[0].strip()
                if dep_name:
                    deps.add(dep_name)
    return deps

def scan_project_for_unaccounted_deps(project_root="."):
    """Scans project, compares with requirements.in, and reports discrepancies."""
    print("\nScanning project for imported modules...")
    py_files = find_python_files(project_root)
    if not py_files:
        print("No Python files found to scan.")
        return

    all_project_imports = set()
    for py_file in py_files:
        all_project_imports.update(extract_imports_from_file(py_file))

    stdlib = get_stdlib_modules()
    project_dependencies = all_project_imports - stdlib

    print(f"Found {len(project_dependencies)} non-stdlib imported modules in .py files: {sorted(list(project_dependencies)) if project_dependencies else 'None'}")

    requirements_in_deps = get_dependencies_from_in_file(REQUIREMENTS_IN_FILE)
    # Normalize for comparison (e.g., scikit-learn in reqs.in -> sklearn in import)
    # This is a common pattern. A more robust solution might involve a mapping.
    normalized_req_in_deps = {dep.replace('-', '_') for dep in requirements_in_deps}


    print(f"Found {len(requirements_in_deps)} dependencies in '{REQUIREMENTS_IN_FILE}': {sorted(list(requirements_in_deps)) if requirements_in_deps else 'None'}")

    unaccounted_for = project_dependencies - normalized_req_in_deps
    # Also check for imports that might map to a differently named package in requirements.in
    # e.g. import PIL -> Pillow in requirements.in
    # This part can get complex, so we'll keep it simple for now.

    if unaccounted_for:
        print("\n--- Potential Missing Dependencies ---")
        print(f"The following modules are imported in your project but not found (or named differently) in '{REQUIREMENTS_IN_FILE}':")
        for dep in sorted(list(unaccounted_for)):
            print(f"  - {dep}")
        print(f"Please review them and add to '{REQUIREMENTS_IN_FILE}' if they are direct project dependencies.")
        print("Note: Some packages have different import names than their PyPI names (e.g., 'sklearn' is 'scikit-learn').")
    else:
        print(f"\nAll non-stdlib imported modules seem to be accounted for in '{REQUIREMENTS_IN_FILE}'.")
    print("--- End of Scan ---")


# --- Existing functions (check_pip_tools_installed, install_pip_tools, etc.) ---
def check_pip_tools_installed():
    """Checks if pip-tools is installed."""
    try:
        subprocess.run(["pip-compile", "--version"], check=True, capture_output=True, text=True, encoding='utf-8')
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_pip_tools():
    """Installs pip-tools."""
    print("pip-tools is not found. Attempting to install it...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pip-tools"], check=True, capture_output=True, text=True, encoding='utf-8')
        print("pip-tools installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing pip-tools: {e.stderr or e.stdout}")
        return False

def create_requirements_in_if_not_exists():
    """Creates a requirements.in file if it doesn't exist, populated with suggested dependencies."""
    if not os.path.exists(REQUIREMENTS_IN_FILE):
        print(f"'{REQUIREMENTS_IN_FILE}' not found.")
        print("Creating a sample one with suggested direct dependencies for your project.")
        print(f"Please review and edit '{REQUIREMENTS_IN_FILE}' to list your project's actual direct dependencies.")
        print("For example, if you need a specific PyTorch version for CUDA 12.1, you might have lines like:")
        print("  torch>=2.2.1 --index-url https://download.pytorch.org/whl/cu121")
        print("  torchvision>=0.17.1 --index-url https://download.pytorch.org/whl/cu121")
        print("  torchaudio>=2.2.1 --index-url https://download.pytorch.org/whl/cu121")
        print("For other packages, just list the name (e.g., 'pandas') to get the latest compatible version.")

        with open(REQUIREMENTS_IN_FILE, "w", encoding='utf-8') as f:
            f.write("# This is your requirements.in file. List your project's direct dependencies here.\n")
            f.write("# pip-compile will use this to generate requirements.txt with all pinned versions.\n")
            f.write("# Example for PyTorch with CUDA 12.1 (adjust as needed):\n")
            f.write("# torch --index-url https://download.pytorch.org/whl/cu121\n")
            f.write("# torchvision --index-url https://download.pytorch.org/whl/cu121\n")
            f.write("# torchaudio --index-url https://download.pytorch.org/whl/cu121\n\n")
            for dep in SUGGESTED_DIRECT_DEPENDENCIES:
                if dep not in ["torch", "torchvision", "torchaudio"]:
                    f.write(f"{dep}\n")
        print(f"\nCreated '{REQUIREMENTS_IN_FILE}'.")
        print(f"Please review and adjust it, then re-run this script.")
        return False
    return True

def compile_requirements(upgrade=False):
    """Compiles requirements.in to requirements.txt using pip-compile."""
    cmd = ["pip-compile", "--resolver=backtracking"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend([REQUIREMENTS_IN_FILE, "-o", REQUIREMENTS_TXT_FILE])

    print(f"\nRunning: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"'{REQUIREMENTS_TXT_FILE}' has been {'updated' if upgrade else 'generated'} successfully.")
        if process.stdout:
            if len(process.stdout.splitlines()) < 20:
                 print("Output from pip-compile:\n", process.stdout)
            else:
                 print(f"pip-compile generated {len(process.stdout.splitlines())} lines of output (not shown for brevity).")
        if process.stderr:
            print("Notices/Warnings from pip-compile:\n", process.stderr)
        print(f"\nNext steps: Install the dependencies using 'pip install -r {REQUIREMENTS_TXT_FILE}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running pip-compile (exit code {e.returncode}):")
        if e.stdout: print(f"--- pip-compile STDOUT ---\n{e.stdout}\n--- END STDOUT ---")
        if e.stderr: print(f"--- pip-compile STDERR ---\n{e.stderr}\n--- END STDERR ---")
        print(f"Failed to {'update' if upgrade else 'generate'} '{REQUIREMENTS_TXT_FILE}'.")
        return False

def main():
    print("Dependency Management Script using pip-tools")
    print("-------------------------------------------")

    if not check_pip_tools_installed():
        if not install_pip_tools():
            print("\nPlease install pip-tools manually (`pip install pip-tools`) and then re-run this script.")
            sys.exit(1)

    if not create_requirements_in_if_not_exists():
        sys.exit(0)

    # Scan for unaccounted dependencies
    scan_project_for_unaccounted_deps(project_root=os.path.dirname(os.path.abspath(__file__)))


    print(f"\nFound '{REQUIREMENTS_IN_FILE}'.")
    while True:
        action = input(f"Do you want to (g)enerate or (u)pgrade '{REQUIREMENTS_TXT_FILE}', (s)can again, or (q)uit? [g/u/s/q]: ").strip().lower()
        if action == 'u':
            compile_requirements(upgrade=True)
            break
        elif action == 'g':
            compile_requirements(upgrade=False)
            break
        elif action == 's':
            scan_project_for_unaccounted_deps(project_root=os.path.dirname(os.path.abspath(__file__)))
            # Continue loop after scan
        elif action == 'q':
            print("Exiting.")
            break
        else:
            print("Invalid action. Please choose 'g', 'u', 's', or 'q'.")

if __name__ == "__main__":
    main()