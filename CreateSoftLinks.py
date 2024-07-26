import os

def create_symbolic_links(source_dir, target_dir):
    # Get a list of all folders in the source directory
    folders = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder)) and not folder.startswith('.git')]

    # Create symbolic links for each folder
    for folder in folders:
        source_path = os.path.join(source_dir, folder)
        target_path = os.path.join(target_dir, folder)

        # Check if the target path already exists
        if os.path.exists(target_path):
            print(f"Symbolic link already exists for {folder}. Skipping.")
        else:
            # Create the symbolic link
            os.symlink(source_path, target_path)
            print(f"Symbolic link created for {folder}.")

if __name__ == "__main__":
    # Get the source directory 
    source_directory = os.getcwd()

    # Get the target directory from the user
    target_directory = input("Enter the path: ")
    # Set the target directory 

    # Call the function to create symbolic links
    create_symbolic_links(source_directory, target_directory)
