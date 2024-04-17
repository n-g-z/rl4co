from functools import lru_cache
from typing import Optional, Union

import torch

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor


def _batchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Same as repeat on dim=0 for Tensordicts as well"""
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])


def batchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    """Same as `einops.repeat(x, 'b ... -> (b r) ...', r=repeats)` but ~1.5x faster and supports TensorDicts.
    Repeats batchify operation `n` times as specified by each shape element.
    If shape is a tuple, iterates over each element and repeats that many times to match the tuple shape.

    Example:
    >>> x.shape: [a, b, c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a*b*c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x


def _unbatchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Undoes batchify operation for Tensordicts as well"""
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))


def unbatchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    """Same as `einops.rearrange(x, '(r b) ... -> b r ...', r=repeats)` but ~2x faster and supports TensorDicts
    Repeats unbatchify operation `n` times as specified by each shape element
    If shape is a tuple, iterates over each element and unbatchifies that many times to match the tuple shape.

    Example:
    >>> x.shape: [a*b*c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a, b, c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(
        shape
    ):  # we need to reverse the shape to unbatchify in the right order
        x = _unbatchify_single(x, s) if s > 0 else x
    return x


def gather_by_index(src, idx, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    return src.gather(dim, idx).squeeze() if squeeze else src.gather(dim, idx)


@torch.jit.script
def get_distance(x: Tensor, y: Tensor):
    """Euclidean distance between two tensors of shape `[..., n, dim]`"""
    return (x - y).norm(p=2, dim=-1)


def get_distances_dm(td: TensorDict, actions: TensorDict):
    """Compute the distance between each pair of consecutive nodes
    in the tour for a batch of tours, based on the distance matrix."""
    batch_size = td["locs"].shape[0]
    # add depot at front
    actions_ordered = torch.cat(
        [torch.zeros(batch_size, 1, dtype=torch.int32, device=td.device), actions],
        dim=1,
    )
    actions_shifted = torch.roll(actions_ordered, -1, dims=-1)
    if "distance_matrix" in td.keys():
        distances = gather_by_index(
            gather_by_index(td["distance_matrix"], actions_ordered, dim=1, squeeze=False),
            actions_shifted,
            dim=2,
            squeeze=False,
        ).squeeze(-1)
    else:
        depot = td["locs"][..., 0:1, :]
        locs_ordered = torch.cat(
            [
                depot,
                gather_by_index(td["locs"], actions).reshape(
                    [batch_size, actions.size(-1), 2]
                ),
            ],
            dim=1,
        )
        ordered_locs_next = torch.roll(locs_ordered, 1, dims=-2)
        distances = get_distance(ordered_locs_next, locs_ordered)
    return distances


def get_total_distance(td: TensorDict, actions: TensorDict):
    """Compute the total tour distance for a batch of tours based on the distance matrix."""
    return get_distances_dm(td, actions).sum(-1)


@torch.jit.script
def get_distances_dm(td: TensorDict, actions: TensorDict):
    """Compute the distance between each pair of consecutive nodes
    in the tour for a batch of tours, based on the distance matrix."""
    batch_size = td["locs"].shape[0]
    # add depot at front
    actions_ordered = torch.cat(
        [torch.zeros(batch_size, 1, dtype=torch.int32, device=td.device), actions],
        dim=1,
    )
    actions_shifted = torch.roll(actions_ordered, -1, dims=-1)
    if "distance_matrix" in td.keys():
        distances = gather_by_index(
            gather_by_index(td["distance_matrix"], actions_ordered, dim=1, squeeze=False),
            actions_shifted,
            dim=2,
            squeeze=False,
        ).squeeze(-1)
    else:
        depot = td["locs"][..., 0:1, :]
        locs_ordered = torch.cat(
            [
                depot,
                gather_by_index(td["locs"], actions).reshape(
                    [batch_size, actions.size(-1), 2]
                ),
            ],
            dim=1,
        )
        ordered_locs_next = torch.roll(locs_ordered, 1, dims=-2)
        distances = get_distance(ordered_locs_next, locs_ordered)
    return distances


@torch.jit.script
def get_total_distance(td: TensorDict, actions: TensorDict):
    """Compute the total tour distance for a batch of tours based on the distance matrix."""
    return get_distances_dm(td, actions).sum(-1)


@torch.jit.script
def get_tour_length(ordered_locs):
    """Compute the total tour distance for a batch of ordered tours.
    Computes the L2 norm between each pair of consecutive nodes in the tour and sums them up.

    Args:
        ordered_locs: Tensor of shape [batch_size, num_nodes, 2] containing the ordered locations of the tour
    """
    ordered_locs_next = torch.roll(ordered_locs, 1, dims=-2)
    return get_distance(ordered_locs_next, ordered_locs).sum(-1)


@torch.jit.script
def get_distance_matrix(locs: Tensor):
    """Compute the euclidean distance matrix for the given coordinates.

    Args:
        locs: Tensor of shape [..., n, dim]
    """
    distance = (locs[..., :, None, :] - locs[..., None, :, :]).norm(p=2, dim=-1)
    return distance


def get_num_starts(td, env_name=None):
    """Returns the number of possible start nodes for the environment based on the action mask"""
    num_starts = td["action_mask"].shape[-1]
    if env_name == "pdp":
        num_starts = (
            num_starts - 1
        ) // 2  # only half of the nodes (i.e. pickup nodes) can be start nodes
    elif env_name in ["cvrp", "sdvrp", "mtsp", "op", "pctsp", "spctsp"]:
        num_starts = num_starts - 1  # depot cannot be a start node

    return num_starts


def select_start_nodes(td, env, num_starts):
    """Node selection strategy as proposed in POMO (Kwon et al. 2020)
    and extended in SymNCO (Kim et al. 2022).
    Selects different start nodes for each batch element

    Args:
        td: TensorDict containing the data. We may need to access the available actions to select the start nodes
        env: Environment may determine the node selection strategy
        num_starts: Number of nodes to select. This may be passed when calling the policy directly. See :class:`rl4co.models.AutoregressiveDecoder`
    """
    num_loc = env.num_loc if hasattr(env, "num_loc") else 0xFFFFFFFF
    if env.name in ["tsp", "atsp"]:
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc
        )
    else:
        # Environments with depot: we do not select the depot as a start node
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc
            + 1
        )
        if env.name == "op":
            if (td["action_mask"][..., 1:].float().sum(-1) < num_starts).any():
                # for the orienteering problem, we may have some nodes that are not available
                # so we need to resample from the distribution of available nodes
                selected = (
                    torch.multinomial(
                        td["action_mask"][..., 1:].float(), num_starts, replacement=True
                    )
                    + 1
                )  # re-add depot index
                selected = rearrange(selected, "b n -> (n b)")
    return selected


def get_best_actions(actions, max_idxs):
    actions = unbatchify(actions, max_idxs.shape[0])
    return actions.gather(0, max_idxs[..., None, None])


def sparsify_graph(cost_matrix: Tensor, k_sparse: Optional[int] = None, self_loop=False):
    """Generate a sparsified graph for the cost_matrix by selecting k edges with the lowest cost for each node.

    Args:
        cost_matrix: Tensor of shape [m, n]
        k_sparse: Number of edges to keep for each node. Defaults to max(n//5, 10) if not provided.
        self_loop: Include self-loop edges in the generated graph when m==n. Defaults to False.
    """
    m, n = cost_matrix.shape
    k_sparse = max(n // 5, 10) if k_sparse is None else k_sparse

    # fill diagonal value with +inf to exclude them from topk results
    if not self_loop and m == n:
        # k_sparse should not exceed n-1 in this occasion
        k_sparse = min(k_sparse, n - 1)
        cost_matrix.fill_diagonal_(torch.inf)

    # select top-k edges with least cost
    topk_values, topk_indices = torch.topk(
        cost_matrix, k=k_sparse, dim=-1, largest=False, sorted=False
    )

    # generate PyG-compatiable edge_index
    edge_index_u = torch.repeat_interleave(
        torch.arange(m, device=cost_matrix.device), topk_indices.shape[1]
    )
    edge_index_v = topk_indices.flatten()
    edge_index = torch.stack([edge_index_u, edge_index_v])

    edge_attr = topk_values.flatten().unsqueeze(-1)
    return edge_index, edge_attr


@lru_cache(5)
def get_full_graph_edge_index(num_node: int, self_loop=False) -> Tensor:
    adj_matrix = torch.ones(num_node, num_node)
    if not self_loop:
        adj_matrix.fill_diagonal_(0)
    edge_index = torch.permute(torch.nonzero(adj_matrix), (1, 0))
    return edge_index
