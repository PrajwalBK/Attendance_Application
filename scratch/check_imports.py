import os
import ast
import sys

def get_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            return []
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split('.')[0])
    return imports

def main():
    root_dir = r"d:\vt_application_v6.2_cpu"
    all_imports = set()
    local_modules = {'core', 'config', 'ui', 'database', 'ui_components'}
    
    for root, dirs, files in os.walk(root_dir):
        if 'venv' in root or '.git' in root or 'scratch' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                imports = get_imports(file_path)
                for imp in imports:
                    if imp and imp not in local_modules:
                        all_imports.add(imp)
                        
    # Filter out standard library modules
    stdlib = set(sys.builtin_module_names)
    # Add common stdlib modules not in builtin_module_names
    common_stdlib = {
        'os', 'sys', 'time', 'datetime', 'json', 'math', 're', 'shutil', 'logging', 
        'subprocess', 'threading', 'queue', 'collections', 'typing', 'functools', 
        'argparse', 'pathlib', 'hashlib', 'uuid', 'socket', 'struct', 'pickle',
        'csv', 'ast', 'inspect', 'traceback', 'warnings', 'copy', 'tempfile',
        'random', 'select', 'multiprocessing', 'concurrent', 'ctypes', 'glob',
        'importlib', 'io', 'abc', 'enum', 'gc', 'platform', 'sqlite3', 'urllib',
        'http', 'base64', 'hmac', 'hashlib', 'ssl', 'weakref', 'xml', 'xmlrpc',
        'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'codecs', 'getpass', 'signal'
    }
    stdlib.update(common_stdlib)
    
    external_imports = all_imports - stdlib
    print("External imports found:")
    for imp in sorted(external_imports):
        print(f"  - {imp}")

if __name__ == '__main__':
    main()
