from math import sqrt
from typing import Optional
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from zmq import device

from rl4co.envs.routing.cvrp import CVRPEnv, CAPACITIES
from rl4co.utils.ops import gather_by_index, get_distance, get_tour_length


class CVRPTWEnv(CVRPEnv):
    """
    An implementation of the Capacitated Vehicle Routing Problem (CVRP) with Time Windows (CVRPTW) environment.
    Inherits from the CVRPEnv class.
    """

    name = "cvrptw"

    def __init__(
        self,
        max_loc: int = 100,  # different default value to CVRPEnv to match max_time, will be scaled
        max_time: int = 480,
        **kwargs,
    ):
        self.min_time = 0  # always 0
        self.max_time = max_time
        super().__init__(max_loc=max_loc, **kwargs)

    def _make_spec(self, td_params: TensorDict):
        super()._make_spec(td_params)

        current_time = UnboundedContinuousTensorSpec(shape=(1), dtype=torch.float32)

        current_loc = UnboundedContinuousTensorSpec(shape=(2), dtype=torch.float32)

        durations = BoundedTensorSpec(
            low=self.min_time,
            high=self.max_time,
            shape=(self.num_loc, 1),
            dtype=torch.int64,
        )

        time_windows = BoundedTensorSpec(
            low=self.min_time,
            high=self.max_time,
            shape=(
                self.num_loc,
                2,
            ),  # each location has a 2D time window (start, end)
            dtype=torch.int64,
        )

        # extend observation specs
        self.observation_spec = CompositeSpec(
            **self.observation_spec,
            current_time=current_time,
            current_loc=current_loc,
            durations=durations,
            time_windows=time_windows,
            # vehicle_idx=vehicle_idx,
        )

    def generate_data(self, batch_size, scale: bool = True) -> TensorDict:
        td = super().generate_data(batch_size)

        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # initialize at zero
        current_time = torch.zeros(*batch_size, 1, dtype=torch.float32)

        ## define service durations
        # generate randomly (first assume service durations of 0, to be changed later)
        durations = torch.zeros(*batch_size, self.num_loc + 1, dtype=torch.float32)

        ## define time windows
        # 1. get distances from depot
        dist = get_distance(td["depot"], td["locs"].transpose(0, 1)).transpose(0, 1)
        dist = torch.cat((torch.zeros(*batch_size, 1), dist), dim=1)
        # 2. randomly create min and max times (as int) for all nodes incl. depot
        min_ts = torch.randint(
            self.min_time, self.max_time, (*batch_size, self.num_loc + 1)
        )
        max_ts = torch.randint(
            self.min_time, self.max_time, (*batch_size, self.num_loc + 1)
        )
        # 3. set the lower value to min, the higher to max
        min_times, max_times = torch.min(min_ts, max_ts), torch.max(min_ts, max_ts)
        # 4. limit min and max times s.t. the distance to the depot is considered
        max_times = torch.min(
            self.max_time - dist, torch.max(max_times, self.min_time + dist + durations)
        )
        min_times = torch.max(
            torch.min(min_times, max_times - durations), self.min_time + dist
        )
        assert torch.all(
            min_times <= max_times
        ), "Please make sure the relation between max_loc and max_time allows for feasible solutions."

        # 5. stack to tensor time_windows
        time_windows = torch.stack((min_times, max_times), dim=-1)
        # 6. Adjust durations
        durations = torch.min(durations, max_times - min_times)
        # 7. time window for the depot
        time_windows[..., 0, 0] = self.min_time
        time_windows[..., 0, 1] = self.max_time

        # for the case later that durations != 0 are used, the durations for the depot must still be 0
        durations[:, 0] = 0.0
        td.update(
            {
                "current_time": current_time,
                "durations": durations,
                "time_windows": time_windows,
            }
        )
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        not_masked = CVRPEnv.get_action_mask(td)
        batch_size = td["locs"].shape[0]
        current_loc = gather_by_index(td["locs"], td["current_node"]).reshape(
            [batch_size, 2]
        )
        dist = get_distance(current_loc, td["locs"].transpose(0, 1)).transpose(0, 1)
        td.update({"current_loc": current_loc, "distances": dist})
        can_reach_in_time = (
            td["current_time"] + td["durations"] + dist <= td["time_windows"][..., 1]
        )
        return not_masked & can_reach_in_time

    def _step(self, td: TensorDict) -> TensorDict:
        batch_size = td["locs"].shape[0]
        # update current_time
        distance = gather_by_index(td["distances"], td["action"]).reshape([batch_size, 1])
        duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
            [batch_size, 1]
        )
        td["current_time"] = (td["action"][:, None] != 0) * (
            torch.max(td["current_time"] + distance, start_times) + duration
        )
        # current_node is updated to the selected action
        td = super()._step(td)
        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]
        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        self.to(td.device)
        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][..., None, :], td["locs"]), -2),
                "demand": td["demand"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=self.device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.vehicle_capacity, device=self.device
                ),
                "visited": torch.zeros(
                    (*batch_size, 1, td["locs"].shape[-2] + 1),
                    dtype=torch.uint8,
                    device=self.device,
                ),
                "current_time": td["current_time"],
                "durations": td["durations"],
                "time_windows": td["time_windows"],
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        """The reward is the negative tour length. Time windows
        are not considered for the calculation of the reward."""
        return super().get_reward(td, actions)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        CVRPEnv.check_solution_validity(td, actions)
        # TODO include for-loop to check time windows

    @staticmethod
    def render(td: TensorDict, actions=None, ax=None, limit_xy: bool = False, **kwargs):
        CVRPEnv.render(td=td, actions=actions, ax=ax, limit_xy=limit_xy, **kwargs)

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand by capacity to be in [0, 1]
        """
        return CVRPEnv.load_data(fpath, batch_size=batch_size)
