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
    max_skill: int = 1
    min_duration: int = 10
    max_duration: int = 30
    system_start_time: float = 0
    system_end_time: float = 480
    # system parameters
    penalty_term: Optional[float] = None
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
    tw_mapping: List[Tuple[float, int, int]] = [
        (0.5, 0, 240),
        (0.5, 240, 480),
    ]  # [(ratio, start, end)]


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
        num_techs = self.params.num_tech

        # (1) Locations
        # depot: we need one per technician to enable compatability with embeddings
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
        # expand to number of technicians
        depot = depot.repeat(1, num_techs, 1)
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
            size=(*batch_size, num_techs, self.params.num_ops),
            device=self.device,
        )
        travel_cost = torch.ones(*batch_size, num_techs, 1)
        # consider ops_mapping
        idx = 0
        for mapping in self.params.tech_mapping:
            techs[:, idx : idx + mapping[0], mapping[1] :] = 0
            travel_cost[:, idx : idx + mapping[0], :] = mapping[2]
            idx += mapping[0]
        # shuffle the technicians skill types
        for ii in range(len(techs)):
            for jj in range(len(techs[ii])):
                tech = techs[ii, jj]
                techs[ii, jj] = tech[torch.randperm(tech.shape[-1])]
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
        # shuffle skills order (on dimension 1)
        perm = torch.randperm(skills.size(1))
        skills = skills[:, perm]
        # add technicians' skills for the depots
        skills = torch.cat([techs, skills], dim=1)

        # (4) Time windows
        # X% of the customers (tw_ratio) require that their operation starts in a given time window (defined in tw_mapping)
        time_windows = torch.cat(
            [
                torch.zeros((*batch_size, self.params.num_loc + num_techs, 1)),
                torch.full(
                    (*batch_size, self.params.num_loc + num_techs, 1), float("inf")
                ),
            ],
            dim=-1,
        )
        if self.params.tw_ratio > 0:
            time_windows[:, :, 0] = self.params.system_start_time
            time_windows[:, :, 1] = self.params.system_end_time
            start = end = num_techs
            for ii, mapping in enumerate(self.params.tw_mapping):
                end += int(self.params.tw_ratio * mapping[0] * self.params.num_loc)
                assert end >= start, "Time window mapping is not valid"
                time_windows[:, start:end, 0] = mapping[1]
                time_windows[:, start:end, 1] = mapping[2]
                start = end

        # (5) service times
        # TODO: add service times

        # (6) penalty term
        if self.params.penalty_term is None:
            penalty_term = torch.full((*batch_size,), float("nan"))
        else:
            penalty_term = torch.full((*batch_size,), self.params.penalty_term)

        td = TensorDict(
            {
                "locs": locs,
                "max_loc": torch.full((*batch_size,), self.params.max_loc),
                "penalty_term": penalty_term,
                "skills": skills,
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
        num_techs = td["travel_cost"].size(1)
        techs, skills = td["skills"][:, :num_techs], td["skills"][:, num_techs:]

        # (1) check skill level
        current_tech_skill = techs[torch.arange(batch_size), td["current_tech"]]
        can_service = (skills <= current_tech_skill.unsqueeze(1).expand_as(skills)).all(
            dim=-1
        )

        # (2) check time windows
        dist = get_distance(
            td["locs"][torch.arange(batch_size), td["current_node"], None],
            td["locs"][:, num_techs:, :],
        )
        can_reach_in_time = (
            td["current_time"][..., None] + dist <= td["time_windows"][..., num_techs:, 1]
        )  # I only need to start the service before the time window ends, not finish it.

        # (3) check if node has been visited
        visited = td["visited"].to(dtype=torch.bool)

        # (4) combine all conditions for customers
        can_visit = can_service & can_reach_in_time & ~visited[:, num_techs:]

        # (5) mask depots/technicians
        in_depot = torch.isin(td["current_node"], torch.arange(num_techs))[..., None]
        already_visited = visited[:, :num_techs]
        visit_depot = ~already_visited * (~in_depot + (can_visit.sum(-1) == 0)[..., None])
        # can visit depot if no other options are available
        visit_depot += (torch.cat((visit_depot, can_visit), dim=-1).sum(-1) == 0)[
            ..., None
        ]
        # however, cannot stay in the same depot as currently
        visit_depot[torch.arange(num_techs), td["current_tech"]] = False

        return torch.cat((visit_depot, can_visit), dim=-1)

    def _step(self, td: TensorDict) -> torch.Tensor:
        """Step function for the Skill-VRP. If a technician returns to the depot, the next technician is sent out.
        The visited node is marked as visited. The reward is set to zero and the done flag is set if all nodes have been visited.
        """
        num_techs = td["travel_cost"].size(1)

        # (1) update current node
        current_node = td["action"]
        current_tech = td["current_tech"]
        current_time = td["current_time"]

        # (2) update technician
        in_depot = torch.isin(current_node, torch.arange(num_techs))
        current_tech = in_depot * current_node + ~in_depot * current_tech

        # (3) update time
        dist = get_distance(
            td["locs"][torch.arange(*td.batch_size), td["current_node"]],
            td["locs"][torch.arange(*td.batch_size), current_node],
        )
        start_times = gather_by_index(td["time_windows"], current_node, dim=1)[..., 0]
        current_time = torch.max(current_time + dist, start_times) * ~in_depot

        # (4) update visited
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, current_node[..., None], 1)

        # SECTION: get done
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "current_tech": current_tech,
                "current_time": current_time,
                "done": done,
                "reward": reward,
                "visited": visited,
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
        visited = torch.zeros(
            size=(*batch_size, td["locs"].shape[-2]),
            dtype=torch.uint8,
            device=self.device,
        )
        visited[:, 0] = 1  # depot/first technician is always visited first

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                # keep some values
                "locs": td["locs"],
                "max_loc": td["max_loc"],
                "penalty_term": td["penalty_term"],
                "skills": td["skills"],
                "time_windows": td["time_windows"],
                "travel_cost": td["travel_cost"],
                # reset others
                "current_time": torch.zeros(
                    (*batch_size,), dtype=torch.float32, device=self.device
                ),
                "current_node": torch.zeros(
                    size=(*batch_size,), dtype=torch.long, device=self.device
                ),
                "current_tech": torch.zeros(
                    size=(*batch_size,), dtype=torch.long, device=self.device
                ),
                "visited": visited,
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        """Calculated the reward, where the reward is the negative total travel cost of the technicians.
        The travel cost depends on the technician and is defined in tech_mapping."""
        # Check that the solution is valid
        if self.check_solution:
            self.check_solution_validity(td, actions)

        num_techs = td["travel_cost"].size(1)

        # (1) Gather dataset in order of tour
        batch_size = td["locs"].shape[0]
        go_from = torch.cat(
            (
                torch.zeros(batch_size, 1),
                actions,
            ),
            dim=1,
        ).to(dtype=torch.int64)
        go_to = torch.roll(go_from, shifts=-1).to(dtype=torch.int64)
        distances = get_distance(
            gather_by_index(td["locs"], go_from, squeeze=False),
            gather_by_index(td["locs"], go_to, squeeze=False),
        )

        # (2) calculate travelling costs depending on the technicians' skill level
        travel_cost = torch.zeros_like(distances)
        current_tech = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        num_routes = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        for ii in range(distances.size(-1)):
            travel_cost[:, ii] = td["travel_cost"][
                torch.arange(batch_size), current_tech
            ].squeeze(-1)
            current_tech += (go_to[:, ii] == 0).int()
            current_tech = current_tech * (current_tech < num_techs).int()
            # count up num_routes if starting a new route (i.e. not staying at depot)
            num_routes += (go_to[:, ii] == 0).int() * (go_from[:, ii] != 0).int()

        # (3) penalty for invalid solutions
        # too many routes (may include further scenarios later on)
        if td["penalty_term"].isnan().any():
            penalty = torch.zeros_like(num_routes)
        else:
            penalty = (
                (num_routes - num_techs)
                * (num_routes > num_techs)
                * td["penalty_term"]
                * td["max_loc"]
            )

        return -(distances * travel_cost).sum(-1) - penalty

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are not visited twice except depot and required skill levels are always met."""
        batch_size = td["skills"].shape[0]
        num_techs = td["travel_cost"].size(1)
        num_cust = td["locs"].shape[1] - num_techs
        technicians = torch.arange(num_techs)

        sorted_actions = actions.data.sort(1).values
        techs, skills = td["skills"][:, :num_techs], td["skills"][:, num_techs:]

        # (1) Check that all nodes (except depot) are visited exactly once
        assert (
            torch.arange(num_techs, num_techs + num_cust, out=sorted_actions.data.new())
            .view(1, -1)
            .expand(batch_size, num_cust)
            == sorted_actions[:, -num_cust:]
        ).all() and torch.isin(
            sorted_actions[:, :-num_cust], technicians
        ).all(), "Invalid tour"

        # (2) make sure all required skill levels are met
        current_tech = torch.zeros(batch_size, dtype=torch.int64, device=td.device)
        skills_ordered = gather_by_index(td["skills"], actions, squeeze=False).to(
            dtype=torch.int64
        )
        tech_skills = torch.zeros_like(skills_ordered)
        techs_temp = torch.zeros_like(actions)
        for ii in range(actions.size(-1)):
            is_tech = torch.isin(actions[:, ii], technicians)
            current_tech = is_tech * actions[:, ii] + ~is_tech * current_tech
            techs_temp[:, ii] = current_tech
            tech_skills[:, ii] = techs[torch.arange(batch_size), current_tech, :]
        assert (skills_ordered <= tech_skills).all(), "Skill level not met"

        # (3) check the number of routes does not exceed the number of technicians
        if td["penalty_term"].isnan().any():
            is_tech = torch.isin(
                torch.cat((torch.zeros(batch_size, 1), actions), dim=1), technicians
            )
            last_index = [
                [batch for batch in is_tech.nonzero() if batch[0] == ii]
                for ii in range(num_techs)
            ]
            last_index = [idx[-1] for idx in last_index]
            for ii in range(num_techs):
                assert (
                    is_tech[ii, : last_index[ii][1]].sum() <= num_techs
                ), "Too many routes"

        # if all checks pass, return True
        return True

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
    from rl4co.utils.ops import get_distance_matrix

    ## --- Rollout and policies --- ##
    def rollout(env, td, policy, max_steps: int = None):
        """Helper function to rollout a policy. Currently, TorchRL does not allow to step
        over envs when done with `env.rollout()`. We need this because for environments that complete at different steps.
        """

        max_steps = float("inf") if max_steps is None else max_steps
        actions = []
        steps = 0

        while not td["done"].all():
            td = policy(td)
            actions.append(td["action"])
            td = env.step(td)["next"]
            steps += 1
            if steps > max_steps:
                print("Max steps reached")
                break
        return torch.stack(actions, dim=1)

    # Simple heuristics (nearest neighbor + capacity check)
    def greedy_policy(td):
        """Select closest available action"""
        available_actions = td["action_mask"]
        # distances
        cost_matrix = get_distance_matrix(locs=td["locs"]).to(dtype=torch.float32)
        current_action = td["current_node"]
        idx_batch = torch.arange(cost_matrix.size(0))
        distances_next = cost_matrix[idx_batch, current_action]
        distances_next[~available_actions.bool()] = float("inf")
        # do not select depot if other actions are available
        distances_next[:, 0] = (
            float("inf") * (available_actions[:, 1:].sum(-1) > 0).float()
        )
        action = torch.argmin(distances_next, dim=-1)
        td.set("action", action)
        return td

    # Totally random policy selecting any available action
    def random_policy(td):
        """Helper function to select a random action from available actions"""
        action = torch.multinomial(td["action_mask"].float(), 1).squeeze(-1)
        td.set("action", action)
        return td

    def run(td, which: str = "random"):
        if which == "greedy":
            actions = rollout(env, td.clone(), random_policy)
        else:
            actions = rollout(env, td.clone(), greedy_policy)

    ## ---END Rollout and policies --- ##

    ## --- Test for feasibility ratio --- ##
    batch_size = 3
    env = SkillVRPEnv(batch_size=[batch_size])

    def run_greedy(td, rewards=[], feasible=[], check_feasibility=True):
        env.check_solution = check_feasibility
        actions = rollout(env, td, greedy_policy)
        reward = env.get_reward(td, actions)
        rewards.append(reward)
        try:
            feasible.append(
                torch.full_like(reward, env.check_solution_validity(td, actions))
            )
        except AssertionError:
            feasible.append(torch.full_like(reward, False))
        return actions, reward

    def run_random(td, rewards=[], feasible=[], check_feasibility: bool = True):
        env.check_solution = check_feasibility
        actions = rollout(env, td, random_policy)
        reward = env.get_reward(td, actions)
        rewards.append(reward)
        try:
            feasible.append(
                torch.full_like(reward, env.check_solution_validity(td, actions))
            )
        except AssertionError:
            feasible.append(torch.full_like(reward, False))
        return actions, reward

    def run_loop(
        env, iterations: int = 100, batch_size: int = 3, check_feasibility: bool = True
    ):
        feasible_rnd, feasible_greedy = [], []
        reward_rnd, reward_greedy = [], []

        print("Start feasibility ratio test")
        for ii in range(iterations):
            print(f"Batch {ii}")
            td = env.reset(batch_size=[batch_size])
            run_greedy(
                td.clone(),
                rewards=reward_greedy,
                feasible=feasible_greedy,
                check_feasibility=check_feasibility,
            )
            run_random(
                td.clone(),
                rewards=reward_rnd,
                feasible=feasible_rnd,
                check_feasibility=check_feasibility,
            )

        reward_rnd = torch.stack(reward_rnd)
        reward_greedy = torch.stack(reward_greedy)
        feasible_rnd = torch.stack(feasible_rnd)
        feasible_greedy = torch.stack(feasible_greedy)

        print("Random policy:")
        print("Mean reward:", reward_rnd.mean().item())
        print("Feasible solutions:", feasible_rnd.mean().item())

        print("Greedy policy:")
        print("Mean reward:", reward_greedy.mean().item())
        print("Feasible solutions:", feasible_greedy.mean().item())
        print()

    # run_loop(env, iterations=10, batch_size=3, check_feasibility=False)

    ## --- END Test for feasibility ratio --- ##

    # actions, reward = run_greedy(
    #     env.reset(batch_size=[batch_size]).clone(), check_feasibility=False
    # )

    ## --- Attention Model --- ##
    from rl4co.models.zoo.am import AttentionModel
    from rl4co.utils.trainer import RL4COTrainer

    batch_size = 3
    env = SkillVRPEnv(batch_size=[batch_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = env.reset(batch_size=[batch_size]).to(device)

    num_skills = td_init["skills"].size(-1)

    model = AttentionModel(
        env,
        baseline="rollout",
        train_data_size=100_000,
        val_data_size=10_000,
        policy_kwargs={"init_embedding_kwargs": {"num_skills": num_skills}},
    ).to(device)

    out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)

    actions_untrained = out["actions"].cpu().detach()
    rewards_untrained = out["reward"].cpu().detach()

    # Plotting
    print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
    for td, actions in zip(td_init, out["actions"].cpu()):
        env.render(td, actions)

    # Training
    trainer = RL4COTrainer(
        max_epochs=3,
        accelerator="gpu",
        devices=1,
        logger=None,
    )
    trainer.fit(model)

    # Testing
    # Greedy rollouts over trained model (same states as previous plot)
    model = model.to(device)
    out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
    actions_trained = out["actions"].cpu().detach()

    # Plotting
    import matplotlib.pyplot as plt

    for i, td in enumerate(td_init):
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        env.render(td, actions_untrained[i], ax=axs[0])
        env.render(td, actions_trained[i], ax=axs[1])
        axs[0].set_title(f"Untrained | Cost = {-rewards_untrained[i].item():.3f}")
        axs[1].set_title(
            r"Trained $\pi_\theta$" + f"| Cost = {-out['reward'][i].item():.3f}"
        )
