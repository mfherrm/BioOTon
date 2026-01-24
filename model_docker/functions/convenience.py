#%% Convenience functions ###
# Method to move tensors to chosen device
def to_device(data, device : str):
    """
        Moves tensors or models (pytorch data) to chosen device.

        Inputs:
            data - the pytorch data to be moved to the specified device

        Output:
            data - the data moved to the specified device
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def trialDir(trial):
    """
        Used to set the ray[tune] runs to a trial id instead of the hyperparameters used.

        Input:
            trial - a ray[tune] trial

        Output:
            str - a formatted directory to save the trial to 
    """
    return f"single_point_RayTune_{trial.trial_id[:6]}"