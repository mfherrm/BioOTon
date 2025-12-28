# Method to move tensors to chosen device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def trialDir(trial):
    return f"/single_point/RayTune/{trial.trial_id[:6]}"