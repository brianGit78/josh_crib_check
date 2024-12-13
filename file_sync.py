import os, subprocess, shutil, secrets, pexpect

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
        #self.remote_path_validation_data = f"{remote_path}/validation_data/"
        self.model_file_path = f"{local_path_root}/crib_model.keras"

    def create_local_directories(self):
        if not os.path.exists(self.local_path_training_data):
            os.makedirs(self.local_path_training_data)
        
        if not os.path.exists(self.local_path_validation_data):
            os.makedirs(self.local_path_validation_data)

    def remove_model_file(self):
        if os.path.exists(self.model_file_path):
            os.remove(self.model_file_path)
            print(f"Removed: {self.model_file_path}")
        else:
            print(f"Model File not found: {self.model_file_path}")

    def start_sync_source(self):
            rsync_cmd = f"rsync -avz --delete  --exclude 'thumbs.db' --exclude 'Thumbs.db' {self.remote_user}@{self.remote_host}:{self.remote_path_training_data} {self.local_path_training_data}"
            try:
                child = pexpect.spawn(rsync_cmd)
                child.expect('password:')
                child.sendline(self.remote_password)
                child.interact()
                print("Directory successfully synchronized.")
            except pexpect.exceptions.ExceptionPexpect as e:
                print("An error occurred while synchronizing directories:", e)

    def split_data_for_validation(self, source_dir, val_dir, val_ratio=0.2):
        """
        Moves a proportion of files from the source directory to the validation directory,
        using a cryptographically secure random selection for higher entropy.
        """
        # Create validation directory if it doesn't exist
        os.makedirs(val_dir, exist_ok=True)

        #delete all the files in the validation directory
        for root, dirs, files in os.walk(val_dir):
            for file in files:
                os.remove(os.path.join(root, file))
                print(f"Removed: {os.path.join(root, file)}")
        
        # Get a list of all files in the source directory
        all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        
        # Calculate number of files to move
        val_count = int(len(all_files) * val_ratio)
        
        # Use secrets.SystemRandom for selecting files
        secure_random = secrets.SystemRandom()
        files_to_move = secure_random.sample(all_files, val_count)
        
        # Move the files
        for f in files_to_move:
            source_path = os.path.join(source_dir, f)
            dest_path = os.path.join(val_dir, f)
            shutil.move(source_path, dest_path)

        print(f"Moved {val_count} files from {source_dir} to {val_dir} with high-entropy randomness.")

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


            
