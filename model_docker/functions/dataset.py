import numpy as np
#%% Dataset functions ###
def splitDataset(dataset, test_split_size : float = 0.2, val_split_size : float = 0.1):
    """
        Splits a dataset into training, test and validation indices.

        Inputs: 

            dataset - a pytorch dataset with a len() method 
            test_split_size : float = 0.2 - portion of the dataset to be used for testing
            val_split_size : float = 0.1 - portion of the dataset to be used for validation

        Outputs:

            train_indices : list[int] - list of indices for training
            test_indices : list[int] - list of indices for testing
            val_indices : list[int] - list of indices for validation 
    """

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    test_split = int(np.floor(test_split_size * dataset_size))
    val_split = int(np.floor(val_split_size * dataset_size))

    val_indices, test_indices, train_indices = indices[:val_split], indices[val_split: (val_split+test_split)], indices[(val_split+test_split):]

    return train_indices, test_indices, val_indices 


def getSamples(split : list, sizes : list = [None, None, None]):
    """
        Computes the indices for the datasets used in the dataloader. Intended to be used after the splitDataset method.

        Inputs:
            split : list - list containing indices of the train, test and val splits
            sizes : list - list containing the number of samples. Uses absolute samples when given an int, percentage for float <= 1 and all sample for None type

        Outputs: 
            samples : list - list containing the split indices
    """
    samples = []
    for idx, data in enumerate(split):
        data_indices = []
        if type(sizes[idx]) is int:
            data_indices = split[idx][:sizes[idx]]
        elif (type(sizes[idx]) is float and sizes[idx] <= 1.0):
            indices = round(len(split[idx])*sizes[idx])
            data_indices = split[idx][:indices]
        elif sizes[idx] is None:
            data_indices = split[idx]
        
        samples.append(data_indices)
        
    return samples