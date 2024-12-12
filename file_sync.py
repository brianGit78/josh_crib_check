import os, subprocess, shutil, secrets

class FileManager:
    def __init__(self, remote_user, remote_password, remote_host, remote_path, local_path_root):
        self.remote_user = remote_user
        self.remote_password = remote_password
        self.remote_host = remote_host
        self.remote_path = remote_path
        self.local_path = local_path_root
        self.local_path_training_data = os.path.join(local_path_root, "training_data")
        self.local_path_validation_data = os.path.join(local_path_root, "validation_data")
        self.remote_path_training_data = f"{remote_path}/training_data/"
        self.remote_path_validation_data = f"{remote_path}/validation_data/"
        self.model_file_path = f"{local_path_root}/crib_model.keras"

    def create_local_directories(self):
        if not os.path.exists(self.local_path_training_data):
            os.makedirs(self.local_path_training_data)
        
        if not os.path.exists(self.local_path_validation_data):
            os.makedirs(self.local_path_validation_data)

    def start_copy(self):
        command = f"scp -r {self.remote_user}@{self.remote_host}:{self.remote_path} {self.local_path}"
        try:
            subprocess.run(command, check=True, shell=True)
            print("Files successfully copied.")
        except subprocess.CalledProcessError as e:
            print("An error occurred while copying files:", e)

    def start_sync_source(self):
        # -a for archive mode (preserves attributes), -v for verbose, -z for compression, --delete to remove files not in source
        command1 = f"rsync -avz --delete {self.remote_user}@{self.remote_host}:{self.remote_path_training_data} {self.local_path_training_data}"
        command2 = f"rsync -avz --delete {self.remote_user}@{self.remote_host}:{self.remote_path_validation_data} {self.local_path_validation_data}"
        try:
            subprocess.run(command1, check=True, shell=True)
            print("Directory successfully synchronized.")
        except subprocess.CalledProcessError as e:
            print("An error occurred while synchronizing directories:", e)

        try:
            subprocess.run(command2, check=True, shell=True)
            print("Directory successfully synchronized.")
        except subprocess.CalledProcessError as e:
            print("An error occurred while synchronizing directories:", e)

    def remove_validation_files_from_training_data(self):
        validation_files = []
        
        # Walk through the validation data directory and collect all file paths
        for root, dirs, files in os.walk(self.local_path_validation_data):
            for file in files:
                validation_files.append(os.path.relpath(os.path.join(root, file), self.local_path_validation_data))
        
        # Remove the collected files from the training data directory
        for file in validation_files:
            training_file_path = os.path.join(self.local_path_training_data, file)
            if os.path.exists(training_file_path):
                os.remove(training_file_path)
                print(f"Removed: {training_file_path}")
            else:
                print(f"File not found: {training_file_path}")

        # remove thumbs.db files
        for root, dirs, files in os.walk(self.local_path_training_data):
            for file in files:
                if file == "Thumbs.db":
                    os.remove(os.path.join(root, file))
                    print(f"Removed: {os.path.join(root, file)}")

        for root, dirs, files in os.walk(self.local_path_validation_data):
            for file in files:
                if file == "Thumbs.db":
                    os.remove(os.path.join(root, file))
                    print(f"Removed: {os.path.join(root, file)}")

    def remove_model_file(self):
        if os.path.exists(self.model_file_path):
            os.remove(self.model_file_path)
            print(f"Removed: {self.model_file_path}")
        else:
            print(f"Model File not found: {self.model_file_path}")
            
