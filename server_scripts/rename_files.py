import paramiko
import getpass

# Connection details
host = 'os-login.lsdf.kit.edu'
port = 22

transport = paramiko.Transport((host, port))

username = input("Enter username: ") or "uyrra"
password = getpass.getpass("Enter password: ")

transport.connect(username = username, password = password)

sftp = paramiko.SFTPClient.from_transport(transport)
sftp.chdir("./data/XenoCanto_denoised_cut")


try:
    files = sftp.listdir("")
    print(files)
    for filename in files:
        if filename.endswith("audio.flac_dn.pt"):
            # 3. Create the new name by replacing '.flac'
            new_name = filename.replace(".flac", "")
            
            print(f"Renaming: {filename} -> {new_name}")
            
            # 4. Execute the rename
            sftp.rename(filename, new_name)

    sftp.close()
    print("Batch rename complete.")

except Exception as e:
    print(f"An error occurred: {e}")