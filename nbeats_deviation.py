import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the file into a pandas DataFrame
df = pd.read_csv('nbeats_smape.csv')

# Set the overall aesthetic for seaborn plots
sns.set(style="whitegrid")

# Convert the 'layer_widths' column to string type
df['layer_widths'] = df['layer_widths'].astype(str)

# List of non-numeric parameters in the new dataset including 'layer_widths' as a string
numeric_params = ['SMAPE', 'in_len', 'out_len', 'lr', 'batch_size', 'dropout', 
                  'expansion_coefficient_dim', 'num_blocks', 'num_layers']
non_numeric_params = ['activation', 'include_hour', 'layer_widths']

# Calculate the number of rows and columns for subplots
num_rows = 4
num_cols = 3

# Create the subplots grid
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Generate boxplots for each numeric parameter
for i, param in enumerate(numeric_params):
    sns.boxplot(ax=axes[i], y=param, data=df, color='skyblue', linewidth=1.5)
    axes[i].set_title(f'Spread of {param}', fontsize=12)
    axes[i].tick_params(axis='both', which='major', labelsize=12)
    axes[i].yaxis.label.set_size(12)

# Generate count plots for each non-numeric parameter
for i, param in enumerate(non_numeric_params):
    if param == 'layer_widths':
        countplot = sns.countplot(ax=axes[i + len(numeric_params)], x=param, data=df)
        axes[i + len(numeric_params)].set_title(f'Count of {param} categories', fontsize=12)
        axes[i + len(numeric_params)].tick_params(axis='both', which='major', labelsize=6)
        axes[i + len(numeric_params)].xaxis.label.set_size(8)
    else:
        countplot = sns.countplot(ax=axes[i + len(numeric_params)], x=param, data=df)
        axes[i + len(numeric_params)].set_title(f'Count of {param} categories', fontsize=12)
        axes[i + len(numeric_params)].tick_params(axis='both', which='major', labelsize=9)
        axes[i + len(numeric_params)].xaxis.label.set_size(9)
    for p in countplot.patches:
        countplot.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=6, xytext=(0, 5), textcoords='offset points')

# Hide the empty subplots
for i in range(len(numeric_params) + len(non_numeric_params), len(axes)):
    axes[i].axis('off')

# Adjust the layout
plt.tight_layout()

# Save the figure to a file
plt.savefig('parameter_spread_nbeats.png', dpi=300)

# Display the plots
plt.show()
