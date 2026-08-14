import os


def get_directory_structure(rootdir, indent=""):
    """Recursively build a text representation of a directory tree."""
    structure = ""
    for entry in sorted(os.listdir(rootdir)):
        path = os.path.join(rootdir, entry)
        if os.path.isdir(path):
            structure += f"{indent}{entry}/\n"
            structure += get_directory_structure(path, indent + "    ")
        else:
            structure += f"{indent}{entry}\n"
    return structure


def save_structure_to_file(rootdir, output_file):
    """Save the directory tree of rootdir to output_file."""
    structure = f"Project Structure:\n{get_directory_structure(rootdir)}"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(structure)


if __name__ == "__main__":
    project_root = os.getcwd()
    output_path = "structure.txt"
    save_structure_to_file(project_root, output_path)
    print(f"Project structure saved to {output_path}")
