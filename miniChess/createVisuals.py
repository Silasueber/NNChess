import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def create_histogram(csv_file):
    # Load CSV file using NumPy
    dataset = np.loadtxt(csv_file, delimiter=',')

    # Extract the last column (assuming it's the target column)
    last_column = dataset[:, -1]
    # last_column = [l - 0.5 for l in last_column]

    # Create a histogram
    plt.hist(last_column, bins=40, color='blue', edgecolor='black')

    # Set plot labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Winning Probability')

    # Show the plot
    plt.show()


# Example usage
create_histogram('data/benni23.csv')

results_list = [
    {'Model': '2 Layer (32,16)', 'Learning Rate': 0.001,
        'Batch Size': 1000, 'Metric Value': -0.0710},
    {'Model': '2 Layer (32,16)', 'Learning Rate': 0.001,
        'Batch Size': 100, 'Metric Value': 0.3331},
    {'Model': '2 Layer (32,16)', 'Learning Rate': 0.001,
        'Batch Size': 10, 'Metric Value': 0.5112},

    {'Model': '2 Layer (32,16)', 'Learning Rate': 0.01,
        'Batch Size': 1000, 'Metric Value': 0.5951},
    {'Model': '2 Layer (32,16)', 'Learning Rate': 0.01,
        'Batch Size': 100, 'Metric Value': 0.5273},
    {'Model': '2 Layer (32,16)', 'Learning Rate': 0.01,
        'Batch Size': 10, 'Metric Value': 0.4456},

    {'Model': '2 Layer (32,16)', 'Learning Rate': 0.1,
        'Batch Size': 1000, 'Metric Value': 0.2911},
    {'Model': '2 Layer (32,16)', 'Learning Rate': 0.1,
        'Batch Size': 100, 'Metric Value': 0.2321},
    {'Model': '2 Layer (32,16)', 'Learning Rate': 0.1,
        'Batch Size': 10, 'Metric Value': 0.3588},

    # 2 Layer (128,64)
    {'Model': '2 Layer (128,64)', 'Learning Rate': 0.001,
        'Batch Size': 1000, 'Metric Value': 0.3074},
    {'Model': '2 Layer (128,64)', 'Learning Rate': 0.001,
        'Batch Size': 100, 'Metric Value': 0.4438},
    {'Model': '2 Layer (128,64)', 'Learning Rate': 0.001,
        'Batch Size': 10, 'Metric Value': 0.6732},

    {'Model': '2 Layer (128,64)', 'Learning Rate': 0.01,
        'Batch Size': 1000, 'Metric Value': 0.7173},
    {'Model': '2 Layer (128,64)', 'Learning Rate': 0.01,
        'Batch Size': 100, 'Metric Value': 0.7568},
    {'Model': '2 Layer (128,64)', 'Learning Rate': 0.01,
        'Batch Size': 10, 'Metric Value': 0.6433},

    {'Model': '2 Layer (128,64)', 'Learning Rate': 0.1,
        'Batch Size': 1000, 'Metric Value': 0.3288},
    {'Model': '2 Layer (128,64)', 'Learning Rate': 0.1,
        'Batch Size': 100, 'Metric Value': 0.5179},
    {'Model': '2 Layer (128,64)', 'Learning Rate': 0.1,
        'Batch Size': 10, 'Metric Value': 0.3626},

    # 3 Layer (256,128,64)
    {'Model': '3 Layer (256,128,64)', 'Learning Rate': 0.001,
        'Batch Size': 1000, 'Metric Value': 0.3380},
    {'Model': '3 Layer (256,128,64)', 'Learning Rate': 0.001,
        'Batch Size': 100, 'Metric Value': 0.6969},
    {'Model': '3 Layer (256,128,64)', 'Learning Rate': 0.001,
        'Batch Size': 10, 'Metric Value': 0.4169},

    {'Model': '3 Layer (256,128,64)', 'Learning Rate': 0.01,
        'Batch Size': 1000, 'Metric Value': 0.9336},
    {'Model': '3 Layer (256,128,64)', 'Learning Rate': 0.01,
        'Batch Size': 100, 'Metric Value': 0.9302},
    {'Model': '3 Layer (256,128,64)', 'Learning Rate': 0.01,
        'Batch Size': 10, 'Metric Value': 0.8010},

    {'Model': '3 Layer (256,128,64)', 'Learning Rate': 0.1,
        'Batch Size': 1000, 'Metric Value': 0.5012},
    {'Model': '3 Layer (256,128,64)', 'Learning Rate': 0.1,
        'Batch Size': 100, 'Metric Value': 0.6848},
    {'Model': '3 Layer (256,128,64)', 'Learning Rate': 0.1,
        'Batch Size': 10, 'Metric Value': 0.6597},

    # 3 Layer (32,24,8)
    {'Model': '3 Layer (32,24,8)', 'Learning Rate': 0.001,
        'Batch Size': 1000, 'Metric Value': 0.2876},
    {'Model': '3 Layer (32,24,8)', 'Learning Rate': 0.001,
        'Batch Size': 100, 'Metric Value': 0.2845},
    {'Model': '3 Layer (32,24,8)', 'Learning Rate': 0.001,
        'Batch Size': 10, 'Metric Value': 0.2845},

    {'Model': '3 Layer (32,24,8)', 'Learning Rate': 0.01,
        'Batch Size': 1000, 'Metric Value': 0.6214},
    {'Model': '3 Layer (32,24,8)', 'Learning Rate': 0.01,
        'Batch Size': 100, 'Metric Value': 0.5795},
    {'Model': '3 Layer (32,24,8)', 'Learning Rate': 0.01,
        'Batch Size': 10, 'Metric Value': 0.4743},

    {'Model': '3 Layer (32,24,8)', 'Learning Rate': 0.1,
        'Batch Size': 1000, 'Metric Value': 0.2990},
    {'Model': '3 Layer (32,24,8)', 'Learning Rate': 0.1,
        'Batch Size': 100, 'Metric Value': 0.3742},
    {'Model': '3 Layer (32,24,8)', 'Learning Rate': 0.1,
        'Batch Size': 10, 'Metric Value': 0.3462},

    # 1 Layer (16)
    {'Model': '1 Layer (16)', 'Learning Rate': 0.001,
        'Batch Size': 1000, 'Metric Value': 0.0054},
    {'Model': '1 Layer (16)', 'Learning Rate': 0.001,
        'Batch Size': 100, 'Metric Value': 0.2991},
    {'Model': '1 Layer (16)', 'Learning Rate': 0.001,
        'Batch Size': 10, 'Metric Value': 0.1945},

    {'Model': '1 Layer (16)', 'Learning Rate': 0.01,
        'Batch Size': 1000, 'Metric Value': 0.5099},
    {'Model': '1 Layer (16)', 'Learning Rate': 0.01,
        'Batch Size': 100, 'Metric Value': 0.4330},
    {'Model': '1 Layer (16)', 'Learning Rate': 0.01,
        'Batch Size': 10, 'Metric Value': 0.4243},

    {'Model': '1 Layer (16)', 'Learning Rate': 0.1,
        'Batch Size': 1000, 'Metric Value': 0.1607},
    {'Model': '1 Layer (16)', 'Learning Rate': 0.1,
        'Batch Size': 100, 'Metric Value': 0.1996},
    {'Model': '1 Layer (16)', 'Learning Rate': 0.1,
        'Batch Size': 10, 'Metric Value': 0.1405},
]

data = [[30, 25, 50, 20],
        [40, 23, 51, 17],
        [35, 22, 45, 19]]
X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X + 0.00, data[0], color='b', width=0.25)
ax.bar(X + 0.25, data[1], color='g', width=0.25)
ax.bar(X + 0.50, data[2], color='r', width=0.25)
