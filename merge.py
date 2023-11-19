import argparse
from stable_baselines3 import PPO
from utils.gitrebasin import naturecnn_permutation_spec, apply_permutation, weight_matching


def weight_averaging(params_a, params_b, inter_param):
    # Get the initial weights from both models
    averaged_state_dict = {}
    assert(inter_param<=1 and inter_param>=0)
    # Average weights
    for key1, value1 in params_a.items():
        #print(key1)
        if key1 in params_b:
            value2 = params_b[key1]
            # Check if the dimensions match
            if value1.size() == value2.size():
                # Compute the average
                averaged_state_dict[key1] = inter_param*value1 + (1-inter_param)*value2
            else:
                raise ValueError("Dimensions of '{}' in state_dict1 and state_dict2 do not match.".format(key1))
        else:
            raise ValueError("Key '{}' not found in state_dict2.".format(key1))
    print("models averaged, inter_param:", inter_param)
    return averaged_state_dict


def gitrebasin(params_a, params_b, inter_param):
    permutation_spec = naturecnn_permutation_spec()
    final_permutation = weight_matching(permutation_spec,
                                        params_a, params_b)
    updated_params = apply_permutation(permutation_spec, final_permutation, params_b)

    layers = ["pi_features_extractor", "vf_features_extractor"]
    sublayers = ["cnn.0", "cnn.2", "cnn.4", "linear.0"]
    # Adjusting all features_extractor layers in sb3 unique model architecture
    for layer in layers:
        for sublayer in sublayers:
            updated_params[f"{layer}.{sublayer}.weight"] = updated_params[f"features_extractor.{sublayer}.weight"]
            updated_params[f"{layer}.{sublayer}.bias"] = updated_params[f"features_extractor.{sublayer}.bias"]

    updated_params = weight_averaging(params_a, updated_params, inter_param)
    return updated_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_a', type=str, required=True)
    parser.add_argument('--model_b', type=str, required=True)
    parser.add_argument('--procedure', type=str, required=True, choices=['avg', 'gitrebasin'], help='specifies merging procedure to merge models a and b')
    parser.add_argument('--save_path', type=str, required=True, help='save location for merged model')
    parser.add_argument('--inter_param', type=float, required=True, help='interpolation parameter (alpha) used in the averaging process')
    return parser.parse_args()


def main():
    args = parse_args()
    print('loading model_a:', args.model_a)
    print('loading model_b:', args.model_b)
    print('procedure:', args.procedure)
    print('inter_param:', args.inter_param)

    model_a = PPO.load(args.model_a, device='cpu')
    model_b = PPO.load(args.model_b, device='cpu')

    params_a = model_a.policy.state_dict()
    params_b = model_b.policy.state_dict()

    updated_params = {}

    if args.procedure == 'avg':
        updated_params = weight_averaging(params_a, params_b, args.inter_param)
    else:
        updated_params = gitrebasin(params_a, params_b, args.inter_param)

    model_b.policy.load_state_dict(updated_params)
    print('saving model to:', args.save_path)
    model_b.save(args.save_path)


if __name__ == '__main__':
    main()

    