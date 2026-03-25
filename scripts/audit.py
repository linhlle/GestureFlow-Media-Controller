import os

def get_dir_size(path='.'):
    total_size = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total_size += entry.stat().st_size
            elif entry.is_dir():
                total_size += get_dir_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total_size

def format_size(size):
    # Converts bytes to Megabytes
    return f"{size / (1024 * 1024):.2f} MB"

def main():
    # Path to your venv site-packages
    # Note: Check if your version is 3.9, 3.12, etc. and update path accordingly
    base_path = 'venv/lib'
    python_dir = os.listdir(base_path)[0] # Usually 'python3.x'
    packages_path = os.path.join(base_path, python_dir, 'site-packages')

    print(f"--- Auditing: {packages_path} ---")
    
    package_sizes = []

    if os.path.exists(packages_path):
        for item in os.listdir(packages_path):
            item_path = os.path.join(packages_path, item)
            if os.path.isdir(item_path):
                size = get_dir_size(item_path)
                package_sizes.append((item, size))
    
    # Sort by size (descending)
    package_sizes.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Package Name':<30} | {'Size':<10}")
    print("-" * 45)
    for name, size in package_sizes[:15]: # Show top 15
        print(f"{name:<30} | {format_size(size)}")

if __name__ == "__main__":
    main()