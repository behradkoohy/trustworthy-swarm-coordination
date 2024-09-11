from pyswarms.base.base_discrete import DiscreteSwarmOptimizer
from pyswarms.backend.generators import create_swarm, generate_discrete_swarm
import pyswarms as ps
import numpy as np
import logging


# Define custom operators
def compute_int_position(swarm, bounds, bh):
    """
    Custom position computation
    """
    try:
        temp_position = swarm.position.copy()
        temp_position += swarm.velocity

        if bounds is not None:
            temp_position = bh(temp_position, bounds)

        # This casting is the only change to the standard operator
        position = temp_position.astype(int)

    except AttributeError:
        print("Please pass a Swarm class")
        raise

    return position


def compute_int_velocity(swarm):
    try:
        swarm_size = swarm.position.shape
        c1 = swarm.options["c1"]
        c2 = swarm.options["c2"]
        w = swarm.options["w"]

        cognitive = (
            c1
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.pbest_pos - swarm.position)
        )
        social = (
            c2 * np.random.uniform(0, 1, swarm_size) * (swarm.best_pos - swarm.position)
        )

        # This casting is the only change to the standard operator
        updated_velocity = ((w * swarm.velocity) + cognitive + social).astype(int)

    except AttributeError:
        print("Please pass a Swarm class")
        raise

    return updated_velocity


# Define a custom topology. This is not 100% necessary, one could also use the
# built-in topologies. The following is the exact same as the Star topology
# but the compute_velocity and compute_position methods have been replaced
# by the custom ones
class IntStar(ps.backend.topology.Topology):
    def __init__(self, static=None, **kwargs):
        super(IntStar, self).__init__(static=True)

    def compute_gbest(self, swarm, **kwargs):
        try:
            if self.neighbor_idx is None:
                self.neighbor_idx = np.tile(
                    np.arange(swarm.n_particles), (swarm.n_particles, 1)
                )
            if np.min(swarm.pbest_cost) < swarm.best_cost:
                best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost)]
                best_cost = np.min(swarm.pbest_cost)
            else:
                best_pos, best_cost = swarm.best_pos, swarm.best_cost

        except AttributeError:
            print("Please pass a Swarm class")
            raise
        else:
            return best_pos, best_cost

    def compute_velocity(self, swarm):
        return compute_int_velocity(swarm)

    def compute_position(self, swarm, bounds, bh):
        return compute_int_position(swarm, bounds, bh)


# Define custom Optimizer class
class IntOptimizerPSO(ps.base.SwarmOptimizer):
    def __init__(self, n_particles, dimensions, options, bounds=None, initpos=None):
        super(IntOptimizerPSO, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=None,
            center=1.0,
            ftol=-np.inf,
            init_pos=initpos,
        )
        self.reset()
        # The periodic strategy will leave the velocities on integer values
        self.bh = ps.backend.handlers.BoundaryHandler(strategy="periodic")
        self.top = IntStar()
        self.rep = ps.utils.Reporter(logger=logging.getLogger(__name__))
        self.name = __name__

    # More or less copy-paste of the optimize method of the GeneralOptimizerPSO
    def optimize(self, func, iters, n_processes=None):
        self.bh.memory = self.swarm.position

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        pool = None if n_processes is None else np.Pool(n_processes)
        for i in self.rep.pbar(iters, self.name):
            self.swarm.current_cost = ps.backend.operators.compute_objective_function(
                self.swarm, func, pool=pool
            )
            self.swarm.pbest_pos, self.swarm.pbest_cost = (
                ps.backend.operators.compute_pbest(self.swarm)
            )
            best_cost_yet_found = self.swarm.best_cost
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, **self.options
            )
            self.rep.hook(best_cost=self.swarm.best_cost)
            # Cou could also just use the custom operators on the next two lines
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm
            )  # compute_int_velocity(self.swarm)
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )  # compute_int_position(self.swarm, self.bounds, self.bh)
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            )
        )
        if n_processes is not None:
            pool.close()

        return final_best_cost, final_best_pos