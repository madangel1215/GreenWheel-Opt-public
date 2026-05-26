"""
Data classes for green energy wheeling allocation optimization.

Sets & Parameters follow the problem formulation:
- I = {1,...,m}  generator plants (PV/wind)
- J = {1,...,n}  consumer sites (corporate buyers)
- T = {1,...,T}  15-min intervals in a billing month
"""

from dataclasses import dataclass
import numpy as np
import yaml
from pathlib import Path

@dataclass
class GeneratorPlant:
    """A renewable energy generator plant.

    Attributes:
        id: Generator index in I.
        capacity_kw: Nameplate capacity G_i^cap.
        technology: 'solar' or 'wind'.
        ppa_price: PPA contract price p_i^ppa (NTD/kWh).
        generation: Generation profile g_i(t), shape (T,), kWh per interval.
    """

    id: int
    capacity_kw: float
    technology: str
    ppa_price: float
    generation: np.ndarray

    @property
    def capacity_factor(self) -> float:
        """Monthly average capacity factor CF_i."""
        max_possible = self.capacity_kw * 0.25 * len(self.generation)  # kW * hours
        return self.generation.sum() / max_possible if max_possible > 0 else 0.0

    @property
    def variability(self) -> float:
        """Generation variability σ_i (coefficient of variation)."""
        nonzero = self.generation[self.generation > 0]
        if len(nonzero) < 2:
            return 0.0
        return float(np.std(nonzero) / np.mean(nonzero))

@dataclass
class ConsumerSite:
    """A corporate consumer site.

    Attributes:
        id: Consumer index in J.
        total_monthly_kwh: Total monthly consumption D_j^total.
        contract_kwh: Contracted green energy volume D_j^contract.
        re100_target: RE100 target ratio r_j^target in [0, 1].
        retail_price: Retail electricity price p_j^retail (NTD/kWh).
        demand: Demand profile d_j(t), shape (T,), kWh per interval.
        consumer_type: 'industrial', 'commercial', or 'hightech'.
    """

    id: int
    total_monthly_kwh: float
    contract_kwh: float
    re100_target: float
    retail_price: float
    demand: np.ndarray
    consumer_type: str = "commercial"

    @property
    def load_factor(self) -> float:
        """Load factor LF_j."""
        peak = self.demand.max()
        avg = self.demand.mean()
        return float(avg / peak) if peak > 0 else 0.0

    @property
    def variability(self) -> float:
        """Demand variability σ_j (coefficient of variation)."""
        if len(self.demand) < 2:
            return 0.0
        return float(np.std(self.demand) / np.mean(self.demand))

    @property
    def contract_coverage(self) -> float:
        """Contract coverage D_j^contract / D_j^total."""
        return self.contract_kwh / self.total_monthly_kwh if self.total_monthly_kwh > 0 else 0.0

@dataclass
class MarketParams:
    """Market and regulatory parameters.

    Attributes:
        surplus_price: Penalty price for surplus electricity (NTD/kWh).
        wheeling_fee: Taipower wheeling fee β^wheel (NTD/kWh).
        fairness_tolerance: Max deviation from proportional allocation α^fair.
        re100_penalty_weight: λ weight for RE100 shortfall in TMI.
    """

    surplus_price: float = 1.0
    wheeling_fee: float = 0.75
    fairness_tolerance: float = 0.15
    re100_penalty_weight: float = 2.0

@dataclass
class WheelingInstance:
    """A complete problem instance for wheeling allocation optimization.

    Contains all generators, consumers, market params, and time-series data
    needed to evaluate an allocation matrix W.
    """

    generators: list[GeneratorPlant]
    consumers: list[ConsumerSite]
    market: MarketParams
    n_intervals: int = 2880  # 96/day * 30 days

    @property
    def m(self) -> int:
        """Number of generator plants."""
        return len(self.generators)

    @property
    def n(self) -> int:
        """Number of consumer sites."""
        return len(self.consumers)

    @property
    def target_proportions(self) -> np.ndarray:
        """Target allocation proportions π_j = D_j^contract / Σ D_j'^contract."""
        contracts = np.array([c.contract_kwh for c in self.consumers])
        total = contracts.sum()
        if total > 0:
            return contracts / total
        return np.ones(self.n) / self.n

    @property
    def total_generation(self) -> float:
        """Total generation across all plants and intervals (kWh)."""
        return sum(g.generation.sum() for g in self.generators)

    @property
    def total_demand(self) -> float:
        """Total demand across all consumers and intervals (kWh)."""
        return sum(c.demand.sum() for c in self.consumers)

    def summary(self) -> dict:
        """Return a summary dict of the instance."""
        return {
            "m_generators": self.m,
            "n_consumers": self.n,
            "n_intervals": self.n_intervals,
            "n_var": self.m * self.n,
            "total_gen_mwh": self.total_generation / 1000,
            "total_demand_mwh": self.total_demand / 1000,
            "gen_demand_ratio": self.total_generation / self.total_demand
            if self.total_demand > 0
            else 0.0,
            "technologies": {
                "solar": sum(1 for g in self.generators if g.technology == "solar"),
                "wind": sum(1 for g in self.generators if g.technology == "wind"),
            },
            "consumer_types": {
                t: sum(1 for c in self.consumers if c.consumer_type == t)
                for t in set(c.consumer_type for c in self.consumers)
            },
        }

@dataclass
class WheelingConfig:
    """Configuration loaded from YAML for the wheeling optimization."""

    # Instance size
    n_generators: int = 5
    n_consumers: int = 10
    n_days: int = 30

    # Generator ranges
    solar_capacity_range_kw: tuple[float, float] = (500, 5000)
    solar_ppa_range: tuple[float, float] = (3.5, 5.0)
    solar_fraction: float = 0.7
    wind_capacity_range_kw: tuple[float, float] = (2000, 10000)
    wind_ppa_range: tuple[float, float] = (5.0, 7.0)

    # Consumer ranges
    demand_peak_range_kw: tuple[float, float] = (100, 2000)
    contract_coverage_range: tuple[float, float] = (0.3, 0.8)
    re100_target_range: tuple[float, float] = (0.5, 1.0)
    retail_price_range: tuple[float, float] = (4.5, 7.5)

    # Market
    surplus_price: float = 1.0
    wheeling_fee: float = 0.75
    fairness_tolerance: float = 0.15
    re100_penalty_weight: float = 2.0

    # Solar profile
    latitude: float = 24.0
    longitude: float = 120.5
    cloud_beta_a: float = 2.0
    cloud_beta_b: float = 5.0

    # Wind profile
    weibull_k: float = 2.0
    weibull_c_range: tuple[float, float] = (7.0, 9.0)
    wind_autocorr: float = 0.95

    # Data source (for semi-real mode)
    data_source: str = "synthetic"  # "synthetic" or "semireal"
    taipower_json_path: str = "data/raw/taipower_37331.json"
    taipower_month: int | None = None  # restrict window to this month (1-12)
    use_real_capacities: bool = True  # use real plant capacities (not synthetic ranges)

    # Optimization
    population_size: int = 100
    n_generations: int = 200
    crossover_prob: float = 0.9
    crossover_eta: float = 20.0
    mutation_eta: float = 20.0
    min_weight_threshold: float = 0.01
    seed: int = 42

    @staticmethod
    def from_yaml(path: str | Path) -> "WheelingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        cfg = WheelingConfig()

        # Instance
        inst = raw.get("instance", {})
        cfg.n_generators = inst.get("n_generators", cfg.n_generators)
        cfg.n_consumers = inst.get("n_consumers", cfg.n_consumers)
        cfg.n_days = inst.get("n_days", cfg.n_days)

        # Generators
        gen = inst.get("generators", {})
        solar = gen.get("solar", {})
        cfg.solar_capacity_range_kw = tuple(
            solar.get("capacity_range_kw", list(cfg.solar_capacity_range_kw))
        )
        cfg.solar_ppa_range = tuple(
            solar.get("ppa_price_range", list(cfg.solar_ppa_range))
        )
        cfg.solar_fraction = solar.get("fraction", cfg.solar_fraction)
        wind = gen.get("wind", {})
        cfg.wind_capacity_range_kw = tuple(
            wind.get("capacity_range_kw", list(cfg.wind_capacity_range_kw))
        )
        cfg.wind_ppa_range = tuple(
            wind.get("ppa_price_range", list(cfg.wind_ppa_range))
        )

        # Consumers
        con = inst.get("consumers", {})
        cfg.demand_peak_range_kw = tuple(
            con.get("demand_peak_range_kw", list(cfg.demand_peak_range_kw))
        )
        cfg.contract_coverage_range = tuple(
            con.get("contract_coverage_range", list(cfg.contract_coverage_range))
        )
        cfg.re100_target_range = tuple(
            con.get("re100_target_range", list(cfg.re100_target_range))
        )
        cfg.retail_price_range = tuple(
            con.get("retail_price_range", list(cfg.retail_price_range))
        )

        # Market
        mkt = inst.get("market", {})
        cfg.surplus_price = mkt.get("surplus_price", cfg.surplus_price)
        cfg.wheeling_fee = mkt.get("wheeling_fee", cfg.wheeling_fee)
        cfg.fairness_tolerance = mkt.get("fairness_tolerance", cfg.fairness_tolerance)
        cfg.re100_penalty_weight = mkt.get(
            "re100_penalty_weight", cfg.re100_penalty_weight
        )

        # Data source
        dat = raw.get("data", {})
        cfg.data_source = dat.get("source", cfg.data_source)
        cfg.taipower_json_path = dat.get("taipower_json_path", cfg.taipower_json_path)
        cfg.taipower_month = dat.get("taipower_month", cfg.taipower_month)
        cfg.use_real_capacities = dat.get("use_real_capacities", cfg.use_real_capacities)

        # Solar
        sol = raw.get("solar", {})
        cfg.latitude = sol.get("latitude", cfg.latitude)
        cfg.longitude = sol.get("longitude", cfg.longitude)
        cfg.cloud_beta_a = sol.get("cloud_cover_beta_a", cfg.cloud_beta_a)
        cfg.cloud_beta_b = sol.get("cloud_cover_beta_b", cfg.cloud_beta_b)

        # Wind
        wnd = raw.get("wind", {})
        cfg.weibull_k = wnd.get("weibull_k", cfg.weibull_k)
        cfg.weibull_c_range = tuple(
            wnd.get("weibull_c_range", list(cfg.weibull_c_range))
        )
        cfg.wind_autocorr = wnd.get("autocorrelation", cfg.wind_autocorr)

        # Optimization
        opt = raw.get("optimization", {})
        cfg.population_size = opt.get("population_size", cfg.population_size)
        cfg.n_generations = opt.get("n_generations", cfg.n_generations)
        cx = opt.get("crossover", {})
        cfg.crossover_prob = cx.get("prob", cfg.crossover_prob)
        cfg.crossover_eta = cx.get("eta", cfg.crossover_eta)
        mut = opt.get("mutation", {})
        cfg.mutation_eta = mut.get("eta", cfg.mutation_eta)
        cfg.min_weight_threshold = opt.get(
            "min_weight_threshold", cfg.min_weight_threshold
        )
        cfg.seed = opt.get("seed", cfg.seed)

        return cfg
