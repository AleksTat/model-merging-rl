"""
Almost all of the code is taken from https://github.com/themrzmaster/git-re-basin-pytorch
Important modification is in lines 56-59
"""
import copy
import torch
from collections import defaultdict
from typing import NamedTuple
from scipy.optimize import linear_sum_assignment


class PermutationSpec(NamedTuple):
  perm_to_axes: dict
  axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
  perm_to_axes = defaultdict(list)
  for wk, axis_perms in axes_to_perm.items():
    for axis, perm in enumerate(axis_perms):
      if perm is not None:
        perm_to_axes[perm].append((wk, axis))
  return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def naturecnn_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None), f"{name}.bias": (p_out, None)}
    dense = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )} if bias else {f"{name}.weight": (p_out, p_in)}

    return permutation_spec_from_axes_to_perm(
    {'features_extractor.cnn.0.weight': ('P_bg0_cnn', None, None, None),
    'features_extractor.cnn.0.bias': ('P_bg0_cnn', None), 
    'features_extractor.cnn.2.weight': ('P_bg1_cnn', 'P_bg0_cnn', None, None),
    'features_extractor.cnn.2.bias': ('P_bg1_cnn', None), 
    'features_extractor.cnn.4.weight': ('P_bg2_cnn', 'P_bg1_cnn', None, None),
    'features_extractor.cnn.4.bias': ('P_bg2_cnn', None), 
    'features_extractor.linear.0.weight': ('P_bg0_linear', 'P_bg2_cnn'), 
    'features_extractor.linear.0.bias': ('P_bg0_linear',), 
    'action_net.weight': (None, 'P_bg0_linear'),
    'action_net.bias': (None, )})


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
  """Get parameter `k` from `params`, with the permutations applied."""
  w = params[k]
  # Don't permute certain layers due to sb3 model architecture (value layer can be permuted for future work)
  if "vf" in k or "pi" in k or "value" in k:
   return w
  for axis, p in enumerate(ps.axes_to_perm[k]):
    # Skip the axis we're trying to permute.
    if axis == except_axis:
      continue
    # None indicates that there is no permutation relevant to that axis.
    if p is not None:
      # Need to "unflatten" our linear layer before permuting it's input weights
      if "linear.0" in k and axis == 1:
        w = w.view(512, 64, 4, 4)
        w = torch.index_select(w, axis, perm[p].int())
        w = w.view(512, -1)
      else:
        w = torch.index_select(w, axis, perm[p].int())

  return w


def apply_permutation(ps: PermutationSpec, perm, params):
  """Apply a `perm` to `params`."""
  return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None):
  """Find a permutation of `params_b` to make them match `params_a`."""
  perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

  perm = {p: torch.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
  perm_names = list(perm.keys())

  for iteration in range(max_iter):
    progress = False
    for p_ix in torch.randperm(len(perm_names)):
      p = perm_names[p_ix]
      n = perm_sizes[p]
      A = torch.zeros((n, n))
      for wk, axis in ps.perm_to_axes[p]:
        w_a = params_a[wk]
        w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
        w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
        w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))
       
        A += w_a @ w_b.T

      ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
      assert (torch.tensor(ri) == torch.arange(len(ri))).all()
      oldL = torch.einsum('ij,ij->i', A, torch.eye(n)[perm[p].long()]).sum()
      newL = torch.einsum('ij,ij->i', A, torch.eye(n)[ci, :]).sum()
      print(f"{iteration}/{p}: {newL - oldL}")
      progress = progress or newL > oldL + 1e-12

      perm[p] = torch.Tensor(ci)

    if not progress:
      break

  return perm
