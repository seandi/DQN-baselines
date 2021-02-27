
import os
import datetime
import matplotlib.pyplot as plt
from typing import Tuple

import numpy as np


def make_dirs(root_dir_name: str, log_dir_name: str, add_run_time: bool = True) -> Tuple[str, str]:
    path = os.path.curdir
    root_dir = os.path.join(path, root_dir_name)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    log_dir = os.path.join(root_dir, log_dir_name)
    if add_run_time:
        date_time = datetime.datetime.now()
        run_time = str(date_time.date()) + "-" + str(date_time.strftime('%H-%M'))
        log_dir = log_dir + "-" + run_time

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        assert not os.listdir(log_dir),\
            "Cannot setup log directory since {0} already exists and is not empty!".format(log_dir)

    model_dir = os.path.join(log_dir, 'models')
    os.mkdir(model_dir)

    return log_dir, model_dir


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


if __name__ == '__main__':
    make_dirs('runs', 'test_log')
