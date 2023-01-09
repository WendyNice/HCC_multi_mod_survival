import pandas as pd
import matplotlib.pyplot as plt

# histogram of different groups
def plot_hist(feature, title):
    plt.hist(feature, color='g', alpha=0.3, label=title, density=1)
    plt.title(title)
    plt.legend()
    # plt.savefig(os.path.join(root_path, title+'_hist.png'))
    # plt.close()
    plt.show()

file_train = '/data/Wendy/HCC/494_8_18.xlsx'
file_valid = '/data/Wendy/HCC/valid_set/label_valid.xlsx'

df_tr = pd.read_excel(file_train)
df_va = pd.read_excel(file_valid)

feature = 'INR'
inr_dt = df_tr[feature].tolist()
inr_va = df_va[feature].tolist()
plot_hist(inr_dt, 'dt')
plot_hist(inr_va, 'va')

