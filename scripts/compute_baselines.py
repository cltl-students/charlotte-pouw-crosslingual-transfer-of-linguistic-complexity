import sys
sys.path.append('C:/Users/charl/Documents/GitHub/charlotte-pouw-crosslingual-transfer-of-linguistic-complexity/preprocessing/')

from sklearn.svm import SVR
import sklearn.metrics as metrics

import argparse
import pandas as pd
import numpy as np
import os, csv

from utils import get_avg_word_length_with_punct, get_avg_token_freq, get_n_low_freq_words, scale


def mean_baseline(df_test, target_label):

    # Get the true values of the target eye-tracking feature (scaled between 0-100)
    y_test = scale(df_test[target_label].tolist())

    # Use the mean as the predicted value every time
    y_pred = len(y_test) * [np.mean(y_test)]

    # Evaluate
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # or mse**(0.5)
    r2 = metrics.r2_score(y_test, y_pred)

    results = {'target': target_label,
               'baseline_model': 'mean',
               'mae': mae,
               'accuracy': (100 - mae),
               'r2': r2,
               'mse': mse,
               'rmse': rmse}

    return results


def svm(df_train, df_test, target_label, setting):

    if setting == 'linguistic':
        target_features = ['lexical_density',                                                            # morpho-syntactic
                           'avg_max_depth', 'avg_links_len', 'max_links_len', 'verbal_head_per_sent']   # syntactic

    elif setting == 'length':
        target_features = ['token_count', 'avg_word_length']

    elif setting == 'frequency':
        target_features = ['avg_token_freq', 'n_low_freq_words']

    elif setting == 'all':
        target_features = ['token_count', 'avg_word_length',                                         # surface
                           'lexical_density',                                                        # morpho-syntactic
                           'avg_max_depth', 'avg_links_len', 'max_links_len', 'verbal_head_per_sent', # syntactic
                           'avg_token_freq', 'n_low_freq_words']                                     # frequency


    X_train = df_train[target_features]
    y_train = scale(df_train[target_label].tolist())
    X_test = df_test[target_features]
    y_test = scale(df_test[target_label])

    # Get predictions from SVR
    svr = SVR(kernel="linear")
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    # Evaluate
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # or mse**(0.5)
    r2 = metrics.r2_score(y_test, y_pred)

    results = {'target': target_label,
               'baseline_model': setting,
               'mae': mae,
               'accuracy': (100-mae),
               'r2': r2,
               'mse': mse,
               'rmse': rmse}

    return results


def compute_baselines_trained_on_geco(args):

    # Create GECO dataframe
    df_train = pd.read_csv(args.geco_path, sep='\t')
    test_language = 'English'

    df_train['avg_token_freq'] = get_avg_token_freq(df_train['text'].tolist(), test_language)
    df_train['n_low_freq_words'] = get_n_low_freq_words(df_train['text'].tolist(), test_language)
    df_train['avg_word_length'] = get_avg_word_length_with_punct(df_train['text'].tolist())

    with open(args.out_path, 'w', encoding='utf8') as outfile:
        outfile.write('test_language,eyetracking-feature,model,mae,accuracy,r2,mse,rmse'+'\n')

        for test_language in os.listdir(args.meco_path):

            # read in MECO test data
            df_test = pd.read_csv(f'{args.meco_path}/test_language/test.tsv', sep='\t')

            # Iterate over target labels
            for target_label in ['scaled_fxc', 'scaled_fpd', 'scaled_tfd', 'scaled_rd']:

                # Calculate four different SVM baselines
                for setting in ['linguistic', 'length', 'sent_length', 'frequency', 'all']:
                    results = svm(df_train, df_test, target_label, setting)
                    outfile.write(f'{test_language},{results["target"]},{results["baseline_model"]},{results["mae"]},{results["accuracy"]},{results["r2"]},{results["mse"]},{results["rmse"]}' + '\n')

                # Also calculate a mean baseline
                mean_results = mean_baseline(df_test, target_label)
                outfile.write(
                    f'{test_language},{mean_results["target"]},{mean_results["baseline_model"]},{mean_results["mae"]},{mean_results["accuracy"]},{mean_results["r2"]},{mean_results["mse"]},{mean_results["rmse"]}' + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geco_path",
        type=str,
        default="'./data/geco/train_data_with_ling_feats.tsv'",
        help="Path to the GECO data with concatenated linguistic features",
    )
    parser.add_argument(
        "--meco_path",
        type=str,
        default="./data/meco/files_per_language",
        help="Path to the MECO data per language",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./results/eye-movement-prediction/baseline_results.csv",
        help="Path to the file where the baseline results will be written",
    )

    args = parser.parse_args()
    compute_baselines_trained_on_geco(args)