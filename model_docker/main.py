import getpass
import os
from pathlib import Path
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
import threading
import torch

from functions.convenience import trialDir
from functions.training import train_model


ray.init(ignore_reinit_error=True)
print("Ray is initialized!")

CUDA_LAUNCH_BLOCKING=1
TORCH_USE_CUDA_DSA = 1

thread_local = threading.local()

#%% Execution ###
if __name__ == '__main__':
    ### SFTP configurations ###
    SFTP_CONFIG = {
        "host" : 'os-login.lsdf.kit.edu',
        "port" : 22,
        "username" : input("Enter username: ") or "uyrra",
        "password" : getpass.getpass("Enter password: ")
    }
    # pool = SFTPConnectionPool(host, port, username , password)

    #### Configurations ####

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # https://medium.com/biased-algorithms/hyperparameter-tuning-with-ray-tune-pytorch-d5749acb314b
    max_epochs = 25
    
    # Define hyperparameter search space
    config = {
        "lr": tune.loguniform(1e-5, 1e-2),          # Learning rate between 1e-5 and 1e-2
        # "batch_size": tune.lograndint(8, 16), # for small batch testing
        "batch_size": tune.lograndint(32, 128),
        "nfft" : tune.choice([512, 1024, 2048, 4096, 8192]),
        # "scale" : tune.uniform(0.5, 4),
        "power" : tune.uniform(0.5, 4),
        "patience" : tune.choice([2, 3, 5, 7, 9]),
        "EarlyDelta" : tune.uniform(0.0015, 0.1),
        "epochs" : tune.randint(15, max_epochs),
        "l1" : tune.loguniform(0.0005, 0.004),
        "l2" : tune.loguniform(0.00075, 0.003),
        "nmels" : tune.randint(64, 256),
        "nmfcc" : tune.randint(32, 128)
        #"optimizer": tune.choice(["adam", "sgd"]),  # Optimizer choice: Adam or SGD
        # "layer_size": tune.randint(64, 256),        # Random integer for layer size (hidden units)
        # "dropout_rate": tune.uniform(0.1, 0.5)      # Dropout rate between 0.1 and 0.5
    }

    # Rayt[tune] parameters
    concurrent_trials = 4
    
    # File locations for the label files
    base_path = "/lsdf01/lsdf/kit/ipf/projects/Bio-O-Ton/Audio_data"
    dawn_file = Path(f"{base_path}/points_single.parquet") # F:/Persönliches/Git/BioOTon/points_single.parquet"
    xeno_file = Path(f"{base_path}/xeno_points_single.parquet") # "F:/Persönliches/Git/BioOTon/xeno_points_single.parquet"
    augmented_file = Path(f"{base_path}/augmented_points_single.parquet") # "F:/Persönliches/Git/BioOTon/augmented_points_single.parquet"

    trainable_with_parameters = tune.with_parameters(
        train_model, sftp_config=SFTP_CONFIG, dataset_files=[dawn_file, xeno_file, augmented_file], spectro_mode = "atls", split = [0.7, 0.1], samples = [0.001,  0.001,  0.001], clip_files = False, clip_length=15 #train_size = int(np.floor(0.7 * len(ds))), val_size = int(np.floor(0.1 * len(ds)))# train_size=500, val_size=100
    )

    cpu_count = int(ray.available_resources().get("CPU", 1))
    gpu_count = int(ray.available_resources().get("GPU", 0))

    trainable_with_resources = tune.with_resources(
        trainable_with_parameters,
        resources={"cpu": int(cpu_count/2)/concurrent_trials, "gpu": 1/concurrent_trials, "accelerator_type:A100":0.5/concurrent_trials}
)

optuna_search = OptunaSearch(
    metric=["loss", "accuracy"],
    mode=["min", "max"]
)

# Currently unused
hyperopt_search = HyperOptSearch(
    metric="loss",
    mode="min",  # Minimize loss
    # points_to_evaluate # Use when some good hyperparameters are known as initial values
)

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='accuracy',
    mode='max',
    max_t=max_epochs,
    grace_period=10,
    brackets=1,
)

# Pass the search algorithm to Ray Tune
tuner = tune.Tuner(
    trainable_with_resources,
    param_space=config,
    # tune_config=tune.TuneConfig(search_alg=hyperopt_search, num_samples=50, trial_dirname_creator=trialDir, max_concurrent_trials=2),
    tune_config=tune.TuneConfig(search_alg=optuna_search, num_samples=50, trial_dirname_creator=trialDir, max_concurrent_trials=concurrent_trials,),
    run_config=tune.RunConfig(storage_path='/home/mherrmann/model_docker/RayResults', name="results")
)
tuner.fit()