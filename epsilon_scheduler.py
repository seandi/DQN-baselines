from typing import Optional
from matplotlib import pyplot as plt


class EpsilonScheduler:
    SCHEDULERS = ['linear', 'exponential']

    def __init__(self,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay_period: Optional[int] = int(2e5),
                 epsilon_decay_factor: Optional[float] = None,
                 schedule: str = 'exponential'
                 ):

        assert schedule in EpsilonScheduler.SCHEDULERS, 'invalid scheduler type, choose either "linear" or "exponential"'

        self.schedule = schedule
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min

        self.epsilon = self.epsilon_start

        if schedule == EpsilonScheduler.SCHEDULERS[0]:
            self.eps_step = (epsilon_start - epsilon_min) / epsilon_decay_period
        elif schedule == EpsilonScheduler.SCHEDULERS[1]:
            self.epsilon_decay_factor = epsilon_decay_factor

    def step(self) -> float:
        """
        Return the epsilon value for the current iteration and
        computes the value for the next iteration.
        :return: eps value for this interaction
        """
        if self.epsilon <= self.epsilon_min:
            return self.epsilon

        old_eps = self.epsilon
        if self.schedule == EpsilonScheduler.SCHEDULERS[0]:
            self.epsilon -= self.eps_step
        elif self.schedule == EpsilonScheduler.SCHEDULERS[1]:
            self.epsilon *= self.epsilon_decay_factor
        else:
            raise NotImplementedError(f"{self.schedule} schedule type not supported yet!")

        return old_eps

    def _plot_epsilon_schedule(self, num_steps:int=int(1e6)) -> None:
        steps = range(1, num_steps+1)

        eps_values = []
        for step in steps:
            eps_values.append(self.step())

        plt.plot(steps, eps_values)

        plt.title('Epsilon schedule')
        plt.xlabel('interaction steps')
        plt.ylabel('epsilon')
        plt.show()

        self.epsilon = self.epsilon_start

if __name__ == '__main__':
    eps_scheduler = EpsilonScheduler(epsilon_start=1.0, epsilon_min=0.02, epsilon_decay_factor=0.999985, schedule='exponential')

    eps_scheduler._plot_epsilon_schedule()

