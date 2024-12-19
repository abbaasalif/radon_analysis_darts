import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# Load the CSV files
df1 = pd.read_csv('nbeats_45.csv')
df2 = pd.read_csv('nbeats_13.csv')
df3 = pd.read_csv('dlinear_45.csv')
df4 = pd.read_csv('dlinear_13.csv')

# Convert 'SyncDate' to datetime
for df in [df1, df2, df3, df4]:
    df['SyncDate'] = pd.to_datetime(df['SyncDate'])

# Specify the line styles for the line plots
line_styles = ['dashed', 'solid', 'dashdot']
colors = ['blue', 'orange', 'green']
labels = ['Actual Filtered', 'Actual', 'Predicted']

# Create a 2x2 grid for the plots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Function to create a subplot
def create_subplot(ax, df, title):
    lines = []
    for i, column in enumerate(['Actual Filtered', 'Actual', 'Predicted']):
        line, = ax.plot(df['SyncDate'], df[column], linestyle=line_styles[i], color=colors[i])
        lines.append(line)
    ax.text(0.5, 0.9, title, fontsize=18, ha='center', va='center', transform=ax.transAxes, alpha=0.5, 
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_ylabel('Radon Flux (pCi/L)')
    return lines

lines1 = create_subplot(axs[0, 0], df1, 'N-BEATS_45')
lines2 = create_subplot(axs[0, 1], df2, 'N-BEATS_13')
lines3 = create_subplot(axs[1, 0], df3, 'D-Linear_45')
lines4 = create_subplot(axs[1, 1], df4, 'D-Linear_13')

# Adjust the layout to be tight
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
# Rotate date labels automatically
fig.autofmt_xdate()
# Add a single legend for all subplots
fig.legend(lines1, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.08))
# Save the figure with a DPI of 300
plt.savefig('2x2_45_13.png', dpi=300)

plt.show()
