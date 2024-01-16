import numpy as np
import matplotlib.pyplot as plt


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

batch_size = 10
learning_rates = [0.001, 0.01, 0.1]
models = ['2 Layer (32,16)', '2 Layer (128,64)',
          '3 Layer (256,128,64)', '3 Layer (32,24,8)', '1 Layer (16)']

# Create a 2-D array to store metric values for each learning rate and model
metric_values_2d = np.zeros((len(learning_rates), len(models)))

# Populate the 2-D array
for i, lr in enumerate(learning_rates):
    for j, model in enumerate(models):
        # Find the entry in results_list for the specific learning rate and model
        entry = next((result['Metric Value'] for result in results_list if result['Learning Rate']
                     == lr and result['Model'] == model and result['Batch Size'] == batch_size), None)

        # If entry is found, assign the metric value to the corresponding position in the array
        if entry is not None:
            metric_values_2d[i, j] = entry

print(metric_values_2d)

X = np.arange(len(models))  # Use the length of models instead of a fixed value
bar_width = 0.2

fig, ax = plt.subplots()
ax.bar(X - bar_width, metric_values_2d[0],
       color='b', width=bar_width, label='LR=0.001')
ax.bar(X, metric_values_2d[1], color='g', width=bar_width, label='LR=0.01')
ax.bar(X + bar_width, metric_values_2d[2],
       color='r', width=bar_width, label='LR=0.1')

# Set labels, title, and legend
ax.set_xticks(X)
ax.set_xticklabels(models)
ax.set_xlabel('Models')
ax.set_ylim(0, 1)
ax.set_ylabel('R2')
ax.set_title('Batchsize = ' + str(batch_size))
ax.legend(title='Learning Rate', loc='upper right')

# Show the plot
plt.show()

metric_v = []
metric_v.append([59, 68, 0, 0])
metric_v.append([40, 32, 0, 0])
metric_v.append([1, 0, 100, 100])

metric_v = np.array(metric_v)
print(metric_v)
X = np.arange(4)  # Use the length of models instead of a fixed value
bar_width = 0.2

fig, ax = plt.subplots()
ax.bar(X - bar_width, metric_v[0],
       color='b', width=bar_width, label='Win')
ax.bar(X, metric_v[1], color='g', width=bar_width, label='Draw')
ax.bar(X + bar_width, metric_v[2],
       color='r', width=bar_width, label='Loss')

# Set labels, title, and legend
ax.set_xticks(X)
ax.set_xlabel('Models')
ax.set_xticklabels(['Random (no check)', 'Random (check)',
                   'Elo 50 (no check)', 'Elo 50 (check)'])
ax.set_ylabel('Frequency')
ax.set_title('Depth = 2')
ax.legend(title='Learning Rate', loc='upper right')

# Show the plot
plt.show()
