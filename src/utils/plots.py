import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime
from egaznepy.visualize import apply_plot_style
# Read the CSV file
now = datetime.datetime.now()

df1 = pd.read_csv('logs/eval/runs/2024-08-07_12-11-58_FINAL_test_with_keele_xcorr/tensorboard/version_0/rca_per_utt.csv', header=None, names=['RCA'])
df2 = pd.read_csv('logs/eval/runs/2024-08-07_12-10-57_FINAL_test_with_fda_xcorr/tensorboard/version_0/rca_per_utt.csv', header=None, names=['RCA'])
df3 = pd.read_csv('logs/eval/runs/2024-08-07_12-06-42_FINAL_test_with_mocha_timit_xcorr/tensorboard/version_0/rca_per_utt.csv', header=None, names=['RCA'])
df4 = pd.read_csv('logs/train/runs/2024-07-31_13-48-47_FINAL_xcorr_with_lpc_train_and_test_with_timit/tensorboard/version_0/rca_per_utt.csv', header=None, names=['RCA'])
df1['Experiment'] = 'keele'
df2['Experiment'] = ' fda'
df3['Experiment'] = 'mocha_timit'
df4['Experiment'] = 'timit'


df = pd.concat([df1, df2, df3, df4])
# Create the violin plot
plt.figure(figsize=(8, 6))
pastel_palette = sns.color_palette("pastel")
sns.violinplot(data=df, x='Experiment', y='RCA', palette=pastel_palette, cut=0)
handles = [plt.Rectangle((0,0),1,1, color=pastel_palette[0], label='Xcorr test with keele'),
           plt.Rectangle((0,0),1,1, color=pastel_palette[1], label='Xcorr test with fda'), 
           plt.Rectangle((0,0),1,1, color=pastel_palette[2], label='Xcorr test with mocha timit'),
           plt.Rectangle((0,0),1,1, color=pastel_palette[3], label='Xcorr test with timit')]
plt.legend(handles=handles)

# Add title and labels
plt.title('RCA Plot')
#plt.xlabel('Experiment')
plt.ylabel('RCA[%] (↑)')
apply_plot_style(0.9)
plt.rcParams['text.usetex'] = False
# Check if results directory exists
if not os.path.exists('results'):
    os.makedirs('results')


plt.savefig('results/violin_plot_{}.pdf'.format(now.strftime("%Y-%m-%d_%H-%M-%S")))#, format='pdf')
plt.show()
