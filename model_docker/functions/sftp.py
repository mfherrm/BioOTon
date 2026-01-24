
import paramiko
import threading
import time
import random

# Initialize the global variable inside the module
_pool = None
_pool_lock = threading.Lock()

class SFTPConnectionPool:
    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.transport = None
        self._lock = threading.Lock()
        self.thread_local = threading.local()

    def _get_transport(self):
        with self._lock:
            # Check if transport is dead or closed
            if self.transport is None or not self.transport.is_active():
                print(f"[{threading.current_thread().name}] Establishing new Transport...")
                self.transport = paramiko.Transport((self.host, self.port))
                self.transport.connect(username=self.username, password=self.password)
            return self.transport

    def get_sftp(self):
        # If the thread already has a client, check if it's still alive
        if hasattr(self.thread_local, "sftp"):
            try:
                self.thread_local.sftp.listdir('.') # Test the connection
                return self.thread_local.sftp
            except:
                del self.thread_local.sftp # It's dead, remove it

        # Attempt to create a new SFTP client with retries
        for i in range(5): # Try 5 times to get a channel
            try:
                time.sleep(random.uniform(0, 32))
                # transport = self._get_transport()
                # sftp = paramiko.SFTPClient.from_transport(transport)
                transport = paramiko.Transport((self.host, self.port))
                transport.connect(username=self.username, password=self.password)
                sftp = paramiko.SFTPClient.from_transport(transport)
                absolute_start_path = "/lsdf01/lsdf/kit/ipf/projects/Bio-O-Ton/Audio_data"
                sftp.chdir(absolute_start_path)
                # self.thread_local.sftp = sftp
                self.thread_local.transport = transport 
                self.thread_local.sftp = sftp
                return sftp
            except Exception as e:
                wait = (i + 1) * 2
                print(f"Channel failed, retrying in {wait}s... Error: {e}")
                time.sleep(wait)
                # Force transport reset on 3rd failure
                if i == 2: self.transport = None 
        
        raise RuntimeError("Could not connect to SFTP after 5 attempts")

def get_pool(host=None, port=None, username=None, password=None):
    """
    Returns a singleton pool. Config is required on the first call.
    """
    global _pool
    with _pool_lock: # Thread-safe initialization
        if _pool is None:
            if not all([host, port, username, password]):
                raise ValueError("SFTP credentials must be provided on the first get_pool call.")
            
            print(f"[{threading.current_thread().name}] Initializing new Connection Pool...")
            _pool = SFTPConnectionPool(host, port, username, password)
    return _pool