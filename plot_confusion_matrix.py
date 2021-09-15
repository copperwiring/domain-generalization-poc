import matplotlib.pyplot as plt
import seaborn as sns


def plot_cm_matrix(gt, pred, cm):
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)  # annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['business', 'health'])
    ax.yaxis.set_ticklabels(['health', 'business'])
    plt.savefig('confusion_matrix.png')
