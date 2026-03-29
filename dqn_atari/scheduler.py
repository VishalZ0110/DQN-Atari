class EpsilonScheduler:
    def __init__(self, min_eps, max_eps, total_steps, exploration_frac=0.1):
        assert min_eps >= 0 and max_eps >= 0 and exploration_frac >= 0
        assert min_eps <= max_eps and exploration_frac < 1 and max_eps <= 1

        self.min_eps = min_eps
        self.max_eps = max_eps
        self.total_steps = total_steps
        self.exploration_frac = exploration_frac

        self.pure_exploration_steps = int(total_steps * exploration_frac)
        self.decay_steps = self.total_steps - self.pure_exploration_steps
        self.decay_rate = (self.max_eps - self.min_eps) / max(1, self.decay_steps)

    def __call__(self, step):
        if step < self.pure_exploration_steps:
            return self.max_eps

        epsilon = self.max_eps - (step - self.pure_exploration_steps) * self.decay_rate
        return max(epsilon, self.min_eps)
