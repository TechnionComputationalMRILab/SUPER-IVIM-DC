import matplotlib.pyplot as plt


def boxplot_ivim(all_data, title):
    labels = ['D*_NET', 'D*_SUPER', 'Dt_NET', 'Dt_SUPER', 'Fp_NET', 'Fp_SUPER']
    fig, ax = plt.subplots()

    # rectangular box plot
    bplot = ax.boxplot(all_data,
                       vert=True,  # vertical box alignment
                       patch_artist=True,  # fill with color
                       labels=labels)  # will be used to label x-ticks
    ax.set_title(title)

    # fill with colors
    colors = ['lightgreen', 'lightblue', 'lightgreen', 'lightblue', 'lightgreen', 'lightblue']

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xlabel('IVIM Parameters')
    ax.set_ylabel('Observed values')
    ax.set_ylim(0, 1.5)
    plt.show()