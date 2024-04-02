import torch

from pydantic import BaseModel
from typing import List, Union

# Environments
# atsp, cvrp, cvrptw, mpdp, mtsp, op, pctsp, pdp, sdvrp, spctsp, svrp, tsp


class BaseEnv(BaseModel):
    name: str
    num_loc: int

    # distribution: Union[str, None]


class VRP(BaseEnv):
    min_loc: float
    max_loc: float


class TSP(VRP):
    name: str = "tsp"
    pass


class ATSP(TSP):
    name: str = "atsp"
    min_dist: float
    max_dist: float
    tmat_class: bool


class PCTSP(TSP):
    name: str = "pctsp"
    pass


class CVRP(VRP):
    name: str = "cvrp"
    min_demand: float
    max_demand: float
    vehicle_capacity: float
    # capacity: float


class CVRPTW(CVRP):
    name: str = "cvrptw"
    max_time: int
    scale: bool


class MPDP(VRP):
    name: str = "mpdp"
    min_num_agents: int
    max_num_agents: int
    objective: str
    check_solution: bool


class MTSP(VRP):
    name: str = "mtsp"
    min_num_agents: int
    max_num_agents: int
    cost_type: str


class OP(VRP):
    name: str = "op"
    max_length: Union[float, torch.Tensor]
    prize_type: str


class PCTSP(TSP):
    name: str = "pctsp"
    penalty_factor: float
    prize_required: float
    check_solution: bool


class PDP(VRP):
    name: str = "pdp"
    pass


class SDVRP(CVRP):
    name: str = "sdvrp"
    pass


class SPCTSP(PCTSP):
    name: str = "spctsp"
    pass


class SkillVRP(VRP):
    name: str = "svrp"
    min_skill: float
    max_skill: float
    tech_costs: List[int]
