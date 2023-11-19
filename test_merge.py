# FILE NOT FINISHED

import argparse
from stable_baselines3 import PPO
from merge import weight_averaging, gitrebasin
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_a', type=str, required=True)
    parser.add_argument('--model_b', type=str, required=True)
    parser.add_argument('--procedure', type=str, required=True)
    return parser.parse_args()


def noway():
    model = PPO.load("/home/beksi/projects/thesis/models/starpilot/starpilot_init0_env0.zip")
    model2 = PPO.load("/home/beksi/projects/thesis/models/starpilot/starpilot_init0_env1.zip")
    updated_params = {}
    updated_params = model.policy.state_dict()
    model2.policy.load_state_dict(updated_params)
    model2.save("./testit")

    
def test_weight_averaging(params_a, inter_param=0.3):
    updated_params = weight_averaging(params_a, params_a, inter_param)
    # Get the keys from both state_dicts
    keys1 = set(params_a.keys())
    keys2 = set(updated_params.keys())

    # Check if the keys are the same
    #if keys1 != keys2:
       # return False

    # Compare the values (tensors) for each key
    """for key in keys1:
        tensor1 = params_a[key]
        tensor2 = updated_params[key]

        # Check if the tensors have the same shape
        if tensor1.shape != tensor2.shape:
            return False

        # Check if the tensors are element-wise equal without any tolerance
        if not torch.equal(tensor1, tensor2):
            return False"""

    # Generate synthetic state_dicts for testing
    state_dict1 = {
        'conv1.weight': torch.randn(3, 3),
        'fc1.weight': torch.randn(10, 5)
    }

    state_dict2 = {
        'conv1.weight': torch.randn(3, 3),
        'fc1.weight': torch.randn(10, 5)
    }

    # Calculate the expected averaged state_dict manually
    expected_averaged_state_dict = {
        'conv1.weight': (inter_param*state_dict1['conv1.weight'] + (1-inter_param)*state_dict2['conv1.weight']),
        'fc1.weight': (inter_param*state_dict1['fc1.weight'] + (1-inter_param)*state_dict2['fc1.weight'])
    }

    # Use your function to calculate the averaged state_dict
    averaged_state_dict = weight_averaging(state_dict1, state_dict2, inter_param)

    # Check if the calculated averaged state_dict matches the expected result
    for key in expected_averaged_state_dict.keys():
        if not torch.equal(averaged_state_dict[key], expected_averaged_state_dict[key]):
            return False

    return True


def test_gitrebasin(params_a, params_b):
    pass


def test_random(params_a, params_b):
    keys1 = set(params_a.keys())
    keys2 = set(params_b.keys())
    for key in keys1:
        tensor1 = params_a[key]
        tensor2 = params_b[key]

        # Check if the tensors have the same shape
        if tensor1.shape != tensor2.shape:
            return False

        # Check if the tensors are element-wise equal without any tolerance
        if not torch.equal(tensor1, tensor2):
            return False
    return True


def main():
    args = parse_args()
    print('loading model_a:', args.model_a)
    print('loading model_b:', args.model_b)
    print('procedure:', args.procedure)

    model_a = PPO.load(args.model_a, device='cpu')
    model_b = PPO.load(args.model_b, device='cpu')

    params_a = model_a.policy.state_dict()
    params_b = model_b.policy.state_dict()

    assert(
        args.procedure == 'noway' or args.procedure == 'gitrebasin'
        ),    f"procedure must be 'avg' or 'gitrebasin', got: {args.procedure}"
    
    if args.procedure == 'noway':
        test_random(params_a, params_b)
    """  if args.procedure == 'avg':
        if test_weight_averaging(params_a):
            print("Weight averaging works correctly")
        else:
            print("Weight averaging does not work correctly")
    elif args.procedure == 'gitrebasin':
        test_gitrebasin(params_a)"""


if __name__ == '__main__':
    main()