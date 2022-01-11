import matplotlib
import matplotlib.pyplot as plt
from IPython import display
matplotlib.use('Qt5Agg')

plt.ion()


def plot(all_scores, all_mean_scores, all_names):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    for i, (scores, mean_scores, name) in enumerate(zip(all_scores, all_mean_scores, all_names)):
        if len(scores) < 1 or len(mean_scores) < 1:
            continue
        columns = 2
        plt.subplot2grid((len(all_scores) // columns, columns),
                         (i // columns, i % columns))
        plt.title(name, fontsize=15 - len(all_scores))
        plt.plot(scores, label="scores")
        plt.plot(mean_scores, label="mean_scores")
        plt.ylim(ymin=0)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1,
                 mean_scores[-1], str(round(mean_scores[-1], 2)))
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.1)
