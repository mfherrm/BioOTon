import paramiko
import getpass

def file_exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False

# Connection details
host = 'os-login.lsdf.kit.edu'
port = 22

transport = paramiko.Transport((host, port))

username = input("Enter username: ") or "uyrra"
password = getpass.getpass("Enter password: ")

transport.connect(username = username, password = password)

sftp = paramiko.SFTPClient.from_transport(transport)
sftp.chdir("./data/AudioTensors_denoised")


try:
    files = sftp.listdir(".")

    rename_files = [f for f in files if f.endswith("_dn.pt")]
    
    print(f"Found {len(rename_files)} files to rename.")
    
    for filename in rename_files:
        new_name = filename.replace("_dn_dn", "_dn")
        new_name = new_name.replace(".pt_dn.pt", "_dn.pt")
        new_name = new_name.replace("\\", "")

        if filename == new_name:
            continue


        if file_exists(sftp, new_name):
            print(f"Skipping: {new_name} already exists.")
            sftp.remove(new_name)
    
            
        print(f"Renaming: {filename} -> {new_name}")
            
        sftp.rename(filename, new_name)

    sftp.close()
    print("Batch rename complete.")

except Exception as e:
    print(f"An error occurred: {e}")