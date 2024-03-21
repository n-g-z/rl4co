from math import sqrt
from typing import Optional
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)

from rl4co.envs.routing.cvrp import CVRPEnv, CAPACITIES
from rl4co.utils.ops import gather_by_index, get_distance_matrix
from rl4co.data.utils import (
    load_npz_to_tensordict,
    load_solomon_instance,
    load_solomon_solution,
)


class CVRPTWEnv(CVRPEnv):
    """Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) environment.
    Inherits from the CVRPEnv class in which capacities are considered.
    Additionally considers time windows within which a service has to be started.

    Args:
        num_loc (int): number of locations (cities) in the VRP, without the depot. (e.g. 10 means 10 locs + 1 depot)
        min_loc (float): minimum value for the location coordinates
        max_loc (float): maximum value for the location coordinates. Defaults to 150.
        min_demand (float): minimum value for the demand of each customer
        max_demand (float): maximum value for the demand of each customer
        max_time (int): maximum time for the environment. Defaults to 480.
        vehicle_capacity (float): capacity of the vehicle
        capacity (float): capacity of the vehicle
        scale (bool): if True, the time windows and service durations are scaled to [0, 1]. Defaults to False.
        td_params: parameters of the environment
    """

    name = "cvrptw"

    def __init__(
        self,
        max_loc: float = 150,  # different default value to CVRPEnv to match max_time, will be scaled
        max_time: int = 480,
        scale: bool = False,
        max_vehicles: int = None,
        lateness_penalty: float = 10.0,
        vehicle_penalty: float = 10.0,
        distance_matrix: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.min_time = 0  # always 0
        self.max_time = max_time
        self.scale = scale
        self.lateness_penalty = lateness_penalty
        self.vehicle_penalty = vehicle_penalty
        self.distance_matrix = distance_matrix
        super().__init__(max_loc=max_loc, **kwargs)
        self.max_vehicles = max_vehicles if max_vehicles is not None else self.num_loc

    def _make_spec(self, td_params: TensorDict):
        super()._make_spec(td_params)

        current_time = UnboundedContinuousTensorSpec(
            shape=(1), dtype=torch.float32, device=self.device
        )

        current_loc = UnboundedContinuousTensorSpec(
            shape=(2), dtype=torch.float32, device=self.device
        )

        # current_vehicle = BoundedTensorSpec(
        #     low=0,
        #     high=self.max_vehicles,
        #     shape=(1),
        #     dtype=torch.int64,
        # )

        durations = BoundedTensorSpec(
            low=self.min_time,
            high=self.max_time,
            shape=(self.num_loc, 1),
            dtype=torch.int64,
            device=self.device,
        )

        time_windows = BoundedTensorSpec(
            low=self.min_time,
            high=self.max_time,
            shape=(
                self.num_loc,
                2,
            ),  # each location has a 2D time window (start, end)
            dtype=torch.int64,
            device=self.device,
        )

        # extend observation specs
        self.observation_spec = CompositeSpec(
            **self.observation_spec,
            current_time=current_time,
            current_loc=current_loc,
            # current_vehicle=current_vehicle,
            durations=durations,
            time_windows=time_windows,
            # vehicle_idx=vehicle_idx,
        )

    def extract_distance_matrix(self, td: TensorDict):
        if "distance_matrix" not in td.keys():
            if self.distance_matrix is not None:
                if self.distance_matrix.shape[0] == td["locs"].shape[0]:
                    td["distance_matrix"] = self.distance_matrix.to(dtype=torch.float32)
                elif self.distance_matrix.shape[0] == 1:
                    td["distance_matrix"] = self.distance_matrix.to(
                        dtype=torch.float32
                    ).expand(td["locs"].shape[0], -1, -1)
            else:
                if "depot" in td.keys():
                    td["distance_matrix"] = get_distance_matrix(
                        locs=torch.cat((td["depot"][..., None, :], td["locs"]), -2)
                    ).to(dtype=torch.float32)
                else:
                    td["distance_matrix"] = get_distance_matrix(locs=td["locs"]).to(
                        dtype=torch.float32
                    )

    def generate_data(self, batch_size) -> TensorDict:
        """
        Generates time windows and service durations for the locations. The depot has a time window of [0, self.max_time].
        The time windows define the time span within which a service has to be started. To reach the depot in time from the last node,
        the end time of each node is bounded by the service duration and the distance back to the depot.
        The start times of the time windows are bounded by how long it takes to travel there from the depot.
        """
        td = super().generate_data(batch_size)

        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        ## define service durations
        # generate randomly (first assume service durations of 0, to be changed later)
        durations = torch.zeros(
            *batch_size, self.num_loc + 1, dtype=torch.float32, device=self.device
        )

        ## define time windows
        # 1. get distances from depot (we don't necessarily have the distance matrix in td yet)
        self.extract_distance_matrix(td)
        dist = td["distance_matrix"][..., 0, :]
        # 2. define upper bound for time windows to make sure the vehicle can get back to the depot in time
        upper_bound = self.max_time - dist - durations
        # 3. create random values between 0 and 1
        ts_1 = torch.rand(*batch_size, self.num_loc + 1, device=self.device)
        ts_2 = torch.rand(*batch_size, self.num_loc + 1, device=self.device)
        # 4. scale values to lie between their respective min_time and max_time and convert to integer values
        min_ts = (dist + (upper_bound - dist) * ts_1).int()
        max_ts = (dist + (upper_bound - dist) * ts_2).int()
        # 5. set the lower value to min, the higher to max
        min_times = torch.min(min_ts, max_ts)
        max_times = torch.max(min_ts, max_ts)
        # 6. reset times for depot
        min_times[..., :, 0] = 0.0
        max_times[..., :, 0] = self.max_time

        # 7. ensure min_times < max_times to prevent numerical errors in attention.py
        # min_times == max_times may lead to nan values in _inner_mha()
        mask = min_times == max_times
        if torch.any(mask):
            min_tmp = min_times.clone()
            min_tmp[mask] = torch.max(
                dist[mask].int(), min_tmp[mask] - 1
            )  # we are handling integer values, so we can simply substract 1
            min_times = min_tmp

            mask = min_times == max_times  # update mask to new min_times
            if torch.any(mask):
                max_tmp = max_times.clone()
                max_tmp[mask] = torch.min(
                    torch.floor(upper_bound[mask]).int(),
                    torch.max(
                        torch.ceil(min_tmp[mask] + durations[mask]).int(),
                        max_tmp[mask] + 1,
                    ),
                )
                max_times = max_tmp

        # scale to [0, 1]
        if self.scale:
            durations = durations / self.max_time
            min_times = min_times / self.max_time
            max_times = max_times / self.max_time
            td["distance_matrix"] = td["distance_matrix"] / self.max_time

        # 8. stack to tensor time_windows
        time_windows = torch.stack((min_times, max_times), dim=-1)

        assert torch.all(
            min_times < max_times
        ), "Please make sure the relation between max_loc and max_time allows for feasible solutions."

        # reset duration at depot to 0
        durations[:, 0] = 0.0
        td.update(
            {
                "durations": durations,
                "time_windows": time_windows,
            }
        )
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """In addition to the constraints considered in the CVRPEnv, the time windows are considered.
        The vehicle can only visit a location if it can reach it in time, i.e. before its time window ends.
        """
        not_masked = CVRPEnv.get_action_mask(td)
        dist = gather_by_index(
            td["distance_matrix"], td["current_node"], dim=1, squeeze=False
        ).squeeze(1)
        can_reach_in_time = (
            td["current_time"] + dist <= td["time_windows"][..., 1]
        )  # I only need to start the service before the time window ends, not finish it.
        last_vehicle = td["current_vehicle"] == td["max_vehicles"] - 1
        # as long as there are unserved nodes, don't allow the last vehicle going back to the depot
        if "done" in td.keys():  # only available after going through _step at least once
            last_vehicle = torch.cat(
                (td["done"], last_vehicle.expand(-1, td["locs"].shape[-2] - 1)), -1
            )
        # if I'm on the last vehicle and the capacity is not enough to serve the remaining customers,
        # go back to the depot to refill, but with the same vehicle, i.e. without resetting current_time
        final_mask = not_masked & (can_reach_in_time | last_vehicle)
        go_back = (final_mask.sum(-1) == 0) & (
            td["visited"].sum(-1) < td["visited"].shape[-1]
        ).squeeze(-1)
        final_mask[go_back, 0] = True
        return final_mask

    def _step(self, td: TensorDict) -> TensorDict:
        """In addition to the calculations in the CVRPEnv, the current time is
        updated to keep track of which nodes are still reachable in time.
        The current_node is updated in the parent class' _step() function.
        """
        batch_size = td["locs"].shape[0]
        # update current_time
        distance = gather_by_index(
            gather_by_index(
                td["distance_matrix"],
                td["current_node"],
                dim=1,
                squeeze=False,
            ),
            td["action"],
            dim=2,
            squeeze=False,
        ).reshape([batch_size, 1])
        duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
            [batch_size, 1]
        )
        td["current_time"] = (
            (td["action"][:, None] != 0)
            | (td["current_vehicle"] == td["max_vehicles"] - 1)
        ) * (torch.max(td["current_time"] + distance, start_times) + duration)
        # current_node is updated to the selected action
        td = super()._step(td)
        # send out next vehicle when going back to the depot
        td["current_vehicle"] += (
            (td["current_node"] == 0) & (td["current_vehicle"] != td["max_vehicles"] - 1)
        ).int()
        return td

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
        reset_distances: bool = False,
    ) -> TensorDict:
        if reset_distances == True:
            self.distance_matrix = None
            if td is not None and "distance_matrix" in td.keys():
                td.pop("distance_matrix")
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]
        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        self.to(td.device)

        # calculate distance matrix, if not available
        self.extract_distance_matrix(td)

        if "depot" in td.keys():
            all_locs = torch.cat((td["depot"][..., None, :], td["locs"]), -2)
        else:
            all_locs = td["locs"]

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
                ),
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=self.device
                ),
                "current_vehicle": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
                ),
                "demand": td["demand"],
                "distance_matrix": td["distance_matrix"],
                "durations": td["durations"],
                "feasible": torch.ones(
                    *batch_size, 1, dtype=torch.bool, device=self.device
                ),
                "locs": all_locs,
                "max_vehicles": torch.tensor(
                    self.max_vehicles, device=self.device
                ).repeat(*batch_size, 1),
                "time_windows": td["time_windows"],
                "used_capacity": torch.zeros((*batch_size, 1), device=self.device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.vehicle_capacity, device=self.device
                ),
                "visited": torch.zeros(
                    (*batch_size, 1, all_locs.shape[-2]),
                    dtype=torch.uint8,
                    device=self.device,
                ),
            },
            batch_size=batch_size,
        )
        td_reset.set(
            "action_mask",
            self.get_action_mask(td_reset),
        )
        return td_reset

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        """The reward is the negative tour length including penalties for lateness
        and for additional routes needed when the vehicle capacity is exceeded."""
        if self.check_solution:
            try:
                self.check_solution_validity(td, actions)
            except AssertionError as e:
                # td["feasible"] = False
                print(e)
        coords = td["locs"]
        batch_size = coords.shape[0]
        actions_ordered = torch.cat(
            [torch.zeros(batch_size, 1, dtype=torch.int32, device=self.device), actions],
            dim=1,
        )
        actions_shifted = torch.roll(actions_ordered, -1, dims=-1)
        distances = gather_by_index(
            gather_by_index(td["distance_matrix"], actions_ordered, dim=1, squeeze=False),
            actions_shifted,
            dim=2,
            squeeze=False,
        ).squeeze(-1)
        current_time = torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)
        total_costs = torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)
        for batch in range(batch_size):
            number_routes = 0
            for step in range(len(actions_ordered[batch])):
                # reset if starting new route and not on last vehicle
                current_time[batch] = (
                    (actions_ordered[batch, step] == 0)
                    & (number_routes < td["max_vehicles"][batch])
                ) * 0.0
                # continue counting
                current_time[batch] = torch.max(
                    current_time[batch] + distances[batch, step],
                    td["time_windows"][batch, actions_shifted[batch, step], 0],
                )
                lateness = torch.max(
                    current_time[batch]
                    - td["time_windows"][batch, actions_shifted[batch, step], 1],
                    torch.zeros_like(current_time[batch]),
                )
                if lateness > 0:  # this check belongs in check_solution_validity
                    td["feasible"][batch] = False
                if (actions_shifted[batch, step] == 0) & (
                    actions_ordered[batch, step] != 0
                ):
                    number_routes += 1
                total_costs[batch] += (
                    distances[batch, step] + lateness * self.lateness_penalty
                )
            total_costs[batch] += (
                torch.max(
                    number_routes - td["max_vehicles"][batch],
                    torch.zeros_like(td["max_vehicles"][batch]),
                )
                * self.vehicle_penalty
            )
            if number_routes > td["max_vehicles"][batch]:
                td["feasible"][batch] = False
        return -total_costs.squeeze(-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        CVRPEnv.check_solution_validity(td, actions)
        batch_size = td["locs"].shape[0]

        # distances to depot
        distances = td["distance_matrix"][..., 0, :]

        # basic checks on time windows
        assert torch.all(distances >= 0.0), "Distances must be non-negative."
        assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
        assert torch.all(
            td["time_windows"][..., :, 0] + distances + td["durations"]
            <= td["time_windows"][..., 0, 1][0]  # max_time is the same for all batches
        ), "vehicle cannot perform service and get back to depot in time."
        assert torch.all(
            td["durations"] >= 0.0
        ), "Service durations must be non-negative."
        assert torch.all(
            td["time_windows"][..., 0] < td["time_windows"][..., 1]
        ), "there are unfeasible time windows"
        # check vehicles can meet deadlines
        curr_time = torch.zeros(batch_size, 1, dtype=torch.float32, device=td.device)
        curr_node = torch.zeros_like(curr_time, dtype=torch.int64, device=td.device)
        for ii in range(actions.size(1)):
            next_node = actions[:, ii]
            dist = gather_by_index(
                gather_by_index(td["distance_matrix"], curr_node, dim=1, squeeze=False),
                next_node,
                dim=2,
                squeeze=False,
            ).reshape([batch_size, 1])
            curr_time = torch.max(
                (curr_time + dist).int(),
                gather_by_index(td["time_windows"], next_node)[..., 0].reshape(
                    [batch_size, 1]
                ),
            )
            assert torch.all(
                curr_time
                <= gather_by_index(td["time_windows"], next_node)[..., 1].reshape(
                    [batch_size, 1]
                )
            ), "vehicle cannot start service before deadline"
            curr_time = curr_time + gather_by_index(td["durations"], next_node).reshape(
                [batch_size, 1]
            )
            curr_node = next_node
            curr_time[curr_node == 0] = 0.0  # reset time for depot
        # TODO check number of routes vs. max_vehicles

    @staticmethod
    def render(td: TensorDict, actions=None, ax=None, scale_xy: bool = False, **kwargs):
        CVRPEnv.render(td=td, actions=actions, ax=ax, scale_xy=scale_xy, **kwargs)

    @staticmethod
    def load_data(
        name: str,
        solomon=False,
        path_instances: str = None,
        type: str = None,
        compute_edge_weights: bool = False,
    ):
        if solomon == True:
            assert type in [
                "instance",
                "solution",
            ], "type must be either 'instance' or 'solution'"
            if type == "instance":
                instance = load_solomon_instance(
                    name=name, path=path_instances, edge_weights=compute_edge_weights
                )
            elif type == "solution":
                instance = load_solomon_solution(name=name, path=path_instances)
            return instance
        return load_npz_to_tensordict(filename=name)

    def extract_from_solomon(self, instance: dict, batch_size: int = 1):
        # extract parameters for the environment from the Solomon instance
        self.min_demand = instance["demand"][1:].min()
        self.max_demand = instance["demand"][1:].max()
        self.vehicle_capacity = instance["capacity"]
        self.min_loc = instance["node_coord"][1:].min()
        self.max_loc = instance["node_coord"][1:].max()
        self.min_time = instance["time_window"][:, 0].min()
        self.max_time = instance["time_window"][:, 1].max()
        if "edge_weight" in instance:
            self.distance_matrix = (
                torch.from_numpy(instance["edge_weight"])
                .expand(batch_size, -1, -1)
                .to(self.device, dtype=torch.float32)
            )
        # assert the time window of the depot starts at 0 and ends at max_time
        assert self.min_time == 0, "Time window of depot must start at 0."
        assert (
            self.max_time == instance["time_window"][0, 1]
        ), "Depot must have latest end time."
        # convert to format used in CVRPTWEnv
        td = TensorDict(
            {
                "locs": torch.tensor(
                    instance["node_coord"],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(batch_size, 1, 1),
                "demand": torch.tensor(
                    instance["demand"][1:],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(batch_size, 1),
                "durations": torch.tensor(
                    instance["service_time"],
                    dtype=torch.int64,
                    device=self.device,
                ).repeat(batch_size, 1),
                "time_windows": torch.tensor(
                    instance["time_window"],
                    dtype=torch.int64,
                    device=self.device,
                ).repeat(batch_size, 1, 1),
            },
            batch_size=batch_size,
        )
        return self.reset(td, batch_size=batch_size)


if __name__ == "__main__":
    from rl4co.models.nn.utils import rollout, random_policy

    device_str = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
            else "cpu"
        )
    )
    device = torch.device(device_str)

    ## Solomon instances
    import vrplib

    names = vrplib.list_names(vrp_type="vrptw")
    ratios = {}

    for name in names:
        if name[0] != "R" or name[1] == "C":
            continue
        print(name)
        env = CVRPTWEnv(scale=False, device=device_str)
        instance = env.load_data(
            name=name, solomon=True, type="instance", compute_edge_weights=True
        )
        sol = env.load_data(name=name, solomon=True, type="solution")
        td = env.extract_from_solomon(instance=instance, batch_size=3).to(device=device)

        reward, _, actions = rollout(
            env=env,
            td=td,
            policy=random_policy,
            max_steps=100_000,
        )
        ratios[name] = td["feasible"].float().mean()
        if len(ratios) > 5:
            break
    print(ratios)
