import seaborn as sns
import matplotlib.pyplot as plt

from utils import get_meco_df

target_feats = ['first_pass_dur', 'fix_count', 'tot_fix_dur', 'tot_regr_from_dur',
                'token_count', 'avg_word_length',
                'lexical_density',
                'avg_max_depth', 'avg_links_len', 'max_links_len', 'verbal_head_per_sent',
                'avg_token_freq', 'n_low_freq_words']

name_mapping = {'first_pass_dur': 'first-pass duration',
                'fix_count': 'fixation count',
                'tot_fix_dur': 'total fixation duration',
                'tot_regr_from_dur': 'regression duration',
                'token_count': 'sentence length (tokens)',
                'avg_word_length': 'avg. word length (characters)',
                'lexical_density': 'lexical density',
                'avg_max_depth': 'parse tree depth',
                'avg_links_len': 'avg. dependency link length',
                'max_links_len': 'max. dependency link length',
                'verbal_head_per_sent': 'number of verbal heads',
                'avg_token_freq': 'avg. word frequency',
                'n_low_freq_words': 'number of low frequency words'}

meco_english = get_meco_df('English')
meco_turkish = get_meco_df('Turkish')
meco_korean = get_meco_df('Korean')

for f in target_feats:
    meco_english = meco_english.rename(columns={f'{f}': name_mapping[f]})
    meco_turkish = meco_turkish.rename(columns={f'{f}': name_mapping[f]})
    meco_korean = meco_korean.rename(columns={f'{f}': name_mapping[f]})

def make_heatmap(df, language):
    corr = df[name_mapping.values()].corr(method='spearman')
    corr = corr.drop(['first-pass duration', 'total fixation duration', 'fixation count', 'regression duration'],
                     axis=0)

    heatmap = sns.heatmap(
        corr[['first-pass duration', 'total fixation duration', 'fixation count', 'regression duration']].sort_values(
            by=['first-pass duration'], ascending=False),
        vmax=1,
        vmin=-0.3,
        cmap='mako_r',
        annot=True,
        linewidths=0,
        linecolor='white')

    x_axis_labels = ['first-pass duration', 'total fixation duration', 'fixation count', 'regression duration']

    heatmap.set_xticklabels(x_axis_labels, rotation=45, horizontalalignment='left')

    heatmap.xaxis.tick_top()  # x axis on top
    heatmap.xaxis.set_label_position('top')

    plt.savefig(f'plots/meco-{language}-heatmap-ling-vs-et.pdf', dpi=300, bbox_inches='tight')

make_heatmap(meco_korean, 'korean')