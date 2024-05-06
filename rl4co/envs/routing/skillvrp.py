from typing import Optional

import torch

from enum import Enum
from pydantic import BaseModel
from typing import List, Tuple
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class Skill(BaseModel):
    type: int
    level: int = 1


class Technician(BaseModel):
    skills: List[Skill]
    travel_cost: float = 1.0


class Customer(BaseModel):
    loc: Tuple[float, float]
    required_skills: List[Skill]
    time_window: Tuple[int, int]
    service_time: int
    precedence: List[Tuple[int, int]] = []


class DepotLoc(Enum):
    center = "center"
    random = "random"
    corner = "corner"


class Instance(BaseModel):
    """
    Attributes needed to define an environment instance.
    To not differentiate between skill levels, simply set max_skill = 1.
    ops_mapping is a list of tuples of shape (a, b, c),
    where a is the number of technicians (int) that can provide b skills (int)
    and have travel costs c (float). -> This modelling does not differentiate travel costs for skill levels yet.
    """

    depot_loc: DepotLoc = DepotLoc.center
    num_loc: int = 20
    num_ops: int = 6
    num_tech: int = 3
    # min-max limits
    min_loc: float = 0.0
    max_loc: float = 150.0
    min_skill: int = 1
    max_skill: int = 10
    min_duration: int = 10
    max_duration: int = 30
    system_start_time: float = 0
    system_end_time: float = 480
    tw_ratio: float = 0.2
    # further definitions
    tech_mapping: List[Tuple[int, int, float]] = [
        (2, 4, 1.0),
        (1, 6, 2.0),
    ]  # [(num_tech, num_ops, travel_cost)]
    cust_mapping: List[Tuple[int, int]] = [
        (10, 1),
        (10, 3),
    ]  # [(num_cust, num_ops)]
    tw_mapping: List[Tuple[float, int, int]] = [(0.5, 0, 240), (0.5, 240, 480)]


class Presets:
    medium_a: int


class SkillVRPEnv(RL4COEnvBase):
    """
    Basic Skill-VRP environment. The environment is a variant of the Capacitated Vehicle Routing Problem (CVRP).
    Each technician has a certain skill-level and each customer node requires a certain skill-level to be serviced.
    Each customer node needs is to be serviced by exactly one technician. Technicians can only service nodes if
    their skill-level is greater or equal to the required skill-level of the node. The environment is episodic and
    the goal is to minimize the total travel cost of the technicians. The travel cost depends on the skill-level of
    the technician. The environment is defined by the following parameters:

    Args:
        num_loc (int): Number of customer locations. Default: 20
        min_loc (float): Minimum value for the location coordinates. Default: 0
        max_loc (float): Maximum value for the location coordinates. Default: 1
        min_skill (float): Minimum skill level of the technicians. Default: 1
        max_skill (float): Maximum skill level of the technicians. Default: 10
        tech_costs (list): List of travel costs for the technicians. Default: [1, 2, 3]. The number of entries in this list determines the number of available technicians.
        td_params (TensorDict): Parameters for the TensorDict. Default: None
    """

    name = "skillvrp"

    def __init__(
        self,
        params: Instance = Instance(),
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # assertions
        assert (
            sum([each[0] for each in params.tech_mapping]) == params.num_tech
        ), "the total number of technicians mapped in ops_mapping (at position 0) must match the number of technicians defined in num_tech"
        assert (
            max([each[1] for each in params.tech_mapping]) <= params.num_ops
        ), "the maximum number of operations mapped in ops_mapping (at position 1) cannot exceed the total number of operations defined in num_ops"
        assert (
            params.min_skill <= params.max_skill
        ), "min_skill cannot be larger than max_skill"
        assert params.min_loc <= params.max_loc, "min_loc cannot be larger than max_loc"
        assert (
            params.min_duration <= params.max_duration
        ), "min_duration cannot be larger than max_duration"
        self.params = params
        # self._make_spec(td_params)

    def _make_spec(self, td_params: TensorDict = None):
        # TODO redo
        """Make the observation and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=self.params.min_loc,
                high=self.params.max_loc,
                shape=(self.params.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            techs=BoundedTensorSpec(
                low=self.params.min_skill,
                high=self.params.max_skill,
                shape=(self.params.num_tech, self.params.num_ops),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            # TODO include time window and service durations
            skill_demand=BoundedTensorSpec(
                low=self.params.min_skill,
                high=self.params.max_skill,
                shape=(self.params.num_loc, 1),
                dtype=torch.float32,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.params.num_loc + 1, 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=self.params.num_loc + 1,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32)
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def generate_data(self, batch_size):
        """Generate data for the basic Skill-VRP. The data consists of the following:
        (1) the locations of the customers and depot,
        (2) the technicians offering services, each with a certain skill type and level,
        (3) the required skills (type and level) of the customers,
        (4) the time windows in which the customers need to be serviced.
        The data is generated randomly within the given bounds."""
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # (1) Locations
        # depot
        if self.params.depot_loc == DepotLoc.center:
            depot = torch.tensor(
                [[self.params.max_loc / 2, self.params.max_loc / 2]]
            ).expand(*batch_size, 1, 2)
        elif self.params.depot_loc == DepotLoc.corner:
            depot = torch.tensor([[0, 0]]).expand(*batch_size, 1, 2)
        else:
            depot = torch.FloatTensor(*batch_size, 1, 2).uniform_(
                self.params.min_loc, self.params.max_loc
            )
        # customer locations
        customer_locs = torch.FloatTensor(*batch_size, self.params.num_loc, 2).uniform_(
            self.params.min_loc, self.params.max_loc
        )
        # # merge locs together
        locs = torch.cat((depot, customer_locs), -2).to(device=self.device)

        # (2) Technicians
        techs = torch.randint(
            low=self.params.min_skill,
            high=self.params.max_skill + 1,  # the interval is not inclusive
            size=(*batch_size, self.params.num_tech, self.params.num_ops),
            device=self.device,
        )
        travel_cost = torch.ones(*batch_size, self.params.num_tech, 1)
        # consider ops_mapping
        idx = 0
        for mapping in self.params.tech_mapping:
            techs[:, idx : idx + mapping[0], mapping[1] :] = 0
            travel_cost[:, idx : idx + mapping[0], :] = mapping[2]
            idx += mapping[0]

        max_skills = torch.max(
            techs, dim=1
        ).values  # this will be the maximum available to the customers

        # (3) Required skills
        skills = (
            torch.rand((*batch_size, self.params.num_loc, self.params.num_ops))
            * (max_skills[:, None, :] - self.params.min_skill)
            + self.params.min_skill
        ).round()
        mask = torch.full_like(skills, False, dtype=torch.bool)
        # consider cust_mapping: how many operations does each customer require
        idx = 0
        for mapping in self.params.cust_mapping:
            slice = skills[:, idx : idx + mapping[0], :]
            random_tensor = torch.rand_like(slice)
            _, indices = torch.topk(random_tensor, mapping[1], dim=-1)
            slice_mask = torch.full_like(slice, False)
            slice_mask.scatter_(dim=2, index=indices, value=True)
            mask[:, idx : idx + mapping[0], :] = slice_mask
            idx += mapping[0]
        skills[~mask] = 0
        # add empty skill for depot
        skills = torch.cat(
            [
                torch.zeros((*batch_size, 1, self.params.num_ops)),
                skills,
            ],
            dim=1,
        )

        # (4) Time windows
        # X% of the customers require that their operation starts in a given time window
        time_windows = torch.cat(
            [
                torch.zeros((*batch_size, self.params.num_loc + 1, 1)),
                torch.full((*batch_size, self.params.num_loc + 1, 1), float("inf")),
            ],
            dim=-1,
        )
        if self.params.tw_ratio > 0:
            time_windows[:, :, 0] = self.params.system_start_time
            time_windows[:, :, 1] = self.params.system_end_time
            start = end = 1
            for i, mapping in enumerate(self.params.tw_mapping):
                end += int(self.params.tw_ratio * mapping[0] * self.params.num_loc)
                assert end >= start, "Time window mapping is not valid"
                time_windows[:, start:end, 0] = mapping[1]
                time_windows[:, start:end, 1] = mapping[2]
                start = end

        td = TensorDict(
            {
                "locs": locs,
                "skills": skills,
                "techs": techs,
                "time_windows": time_windows,
                "travel_cost": travel_cost,
            },
            batch_size=batch_size,
            device=self.device,
        )
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Calculates the action mask for the Skill-VRP. The action mask is a binary mask that indicates which customer nodes can be services, given the previous decisions.
        For the Skill-VRP, a node can be serviced if the technician has the required skill-level and the node has not been visited yet.
        The depot cannot be visited if there are still unserved nodes and the technician either just visited the depot or is the last technician
        (because every customer node needs to be visited).
        """
        batch_size = td["locs"].shape[0]
        dist = get_distance(
            td["locs"][torch.arange(batch_size), td["current_node"]],
            td["locs"].transpose(0, 1),
        ).transpose(0, 1)

        # (1) check skill level
        current_tech_skill = td["techs"][torch.arange(batch_size), td["current_tech"]]
        can_service = (
            td["skills"] <= current_tech_skill.unsqueeze(1).expand_as(td["skills"])
        ).all(dim=-1)
        # (2) check time windows
        can_reach_in_time = (
            td["current_time"] + dist <= td["time_windows"][..., 1]
        )  # I only need to start the service before the time window ends, not finish it.

        # (3) check if node has been visited
        visited = td["visited"].to(can_service.dtype)

        # (4) combine all conditions
        can_visit = can_service & can_reach_in_time & ~visited.squeeze(-1)

        # (5) mask depot
        can_visit[:, 0] = ~((td["current_node"] == 0) & (can_visit[:, 1:].sum(-1) > 0))

        return can_visit

    def _step(self, td: TensorDict) -> torch.Tensor:
        """Step function for the Skill-VRP. If a technician returns to the depot, the next technician is sent out.
        The visited node is marked as visited. The reward is set to zero and the done flag is set if all nodes have been visited.
        """
        current_node = td["action"][:, None]  # Add dimension for step

        # if I go back to the depot, send out next technician
        td["current_tech"] += (current_node == 0).int()

        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-2, current_node[..., None], 1)

        # SECTION: get done
        done = visited.sum(-2) == visited.size(-2)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[0]
        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        self.to(td.device)

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                # keep some values
                "locs": td["locs"],
                "skills": td["skills"],
                "techs": td["techs"],
                "time_windows": td["time_windows"],
                "travel_cost": td["travel_cost"],
                # reset others
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=self.device
                ),
                "current_node": torch.zeros(
                    size=(*batch_size,), dtype=torch.long, device=self.device
                ),
                "current_tech": torch.zeros(
                    size=(*batch_size,), dtype=torch.long, device=self.device
                ),
                "visited": torch.zeros(
                    size=(*batch_size, td["locs"].shape[-2], 1),
                    dtype=torch.uint8,
                    device=self.device,
                ),
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        """Calculated the reward, where the reward is the negative total travel cost of the technicians.
        The travel cost depends on the skill-level of the technician."""
        # Check that the solution is valid
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # Gather dataset in order of tour
        batch_size = td["locs"].shape[0]
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

        # calculate travelling costs depending on the technicians' skill level
        costs = torch.zeros(batch_size, locs_ordered.size(-2), device=self.device)
        indices = torch.nonzero(actions == 0)
        start = tech = 0
        batch = 0
        for each in indices:
            if each[0] > batch:
                costs[batch, start:] = self.tparams.ech_costs[tech]
                start = tech = 0
                batch = each[0]
            end = (
                each[-1] + 1
            )  # indices in locs_ordered are shifted by one due to added depot in the front
            costs[batch, start:end] = self.params.tech_costs[tech]
            tech += 1
            start = end
        costs[batch, start:] = self.params.tech_costs[tech]

        travel_to = torch.roll(locs_ordered, -1, dims=-2)
        distances = get_distance(locs_ordered, travel_to)
        return -(distances * costs).sum(-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are not visited twice except depot and required skill levels are always met."""
        batch_size, graph_size = td["skills"].shape[0], td["skills"].shape[1]
        sorted_pi = actions.data.sort(1).values

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, graph_size)
            == sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # make sure all required skill  levels are met
        indices = torch.nonzero(actions == 0)
        skills = torch.cat(
            [torch.zeros(batch_size, 1, 1, device=td.device), td["skills"]], 1
        )
        skills_ordered = gather_by_index(skills, actions).reshape(
            [batch_size, actions.size(-1), 1]
        )
        batch = start = tech = 0
        for each in indices:
            if each[0] > batch:
                start = tech = 0
                batch = each[0]
            assert (
                skills_ordered[batch, start : each[1]] <= td["techs"][batch, tech]
            ).all(), "Skill level not met"
            start = each[1] + 1  # skip the depot
            tech += 1

    @staticmethod
    def render(
        td: TensorDict,
        actions=None,
        ax=None,
        **kwargs,
    ):
        import matplotlib.pyplot as plt
        import numpy as np

        from matplotlib import cm, colormaps

        num_routine = (actions == 0).sum().item() + 2
        base = colormaps["nipy_spectral"]
        color_list = base(np.linspace(0, 1, num_routine))
        cmap_name = base.name + str(num_routine)
        out = base.from_list(cmap_name, color_list, num_routine)

        if ax is None:
            # Create a plot of the nodes
            _, ax = plt.subplots()

        td = td.detach().cpu()

        if actions is None:
            actions = td.get("action", None)

        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]
            actions = actions[0]

        locs = td["locs"]

        # add the depot at the first action and the end action
        actions = torch.cat([torch.tensor([0]), actions, torch.tensor([0])])

        # gather locs in order of action if available
        if actions is None:
            log.warning("No action in TensorDict, rendering unsorted locs")
        else:
            locs = locs

        # Cat the first node to the end to complete the tour
        x, y = locs[:, 0], locs[:, 1]

        # plot depot
        ax.scatter(
            locs[0, 0],
            locs[0, 1],
            edgecolors=cm.Set2(2),
            facecolors="none",
            s=100,
            linewidths=2,
            marker="s",
            alpha=1,
        )

        # plot visited nodes
        ax.scatter(
            x[1:],
            y[1:],
            edgecolors=cm.Set2(0),
            facecolors="none",
            s=50,
            linewidths=2,
            marker="o",
            alpha=1,
        )

        # text depot
        ax.text(
            locs[0, 0],
            locs[0, 1] - 0.025,
            "Depot",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=10,
            color=cm.Set2(2),
        )

        # plot actions
        color_idx = 0
        for action_idx in range(len(actions) - 1):
            if actions[action_idx] == 0:
                color_idx += 1
            from_loc = locs[actions[action_idx]]
            to_loc = locs[actions[action_idx + 1]]
            ax.plot(
                [from_loc[0], to_loc[0]],
                [from_loc[1], to_loc[1]],
                color=out(color_idx),
                lw=1,
            )
            ax.annotate(
                "",
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(arrowstyle="-|>", color=out(color_idx)),
                size=15,
                annotation_clip=False,
            )
        plt.show()


if __name__ == "__main__":
    env = SkillVRPEnv(batch_size=[3])
    td = env.reset()
    action_mask = env.get_action_mask(td)
    print("action_mask: ", action_mask)
