import torch
from modules.grasr import GraSR
from modules.simlearner import SimLearner
from modules.deepfold import DeepFold


def get_model(args, device, model_path=''):
    model_name = args.model_name
    if model_name == 'simlearner':
        model = SimLearner(args)
    elif model_name == 'grasr':
        model = GraSR()
    elif model_name == 'deepfold':
        model = DeepFold()
    model = model.to(device)
    
    print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    
    if model_path != '':
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model