import os, shutil, secrets, pexpect

class FileManager:
    def __init__(self, model_name):
        self.local_path = os.getcwd()
        self.local_path_training_data = os.path.join(self.local_path, "training_data")
        self.local_path_validation_data = os.path.join(self.local_path, "validation_data")
        self.model_file_path = os.path.join(self.local_path, model_name)
        
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

    def sync_source(self, remote_user, remote_password, remote_host, remote_path):
        rsync_cmd = f"rsync -avz --delete  --exclude 'thumbs.db' --exclude 'Thumbs.db' {remote_user}@{remote_host}:{remote_path}/ {self.local_path_training_data}/"
        try:
            child = pexpect.spawn(rsync_cmd)
            child.expect('password:')
            child.sendline(remote_password)
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
