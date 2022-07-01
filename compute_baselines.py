import argparse

from sklearn.svm import SVR
import sklearn.metrics as metrics

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
    df_train = pd.read_csv(f'data/geco/english/train_scaled_and_correct_feats.tsv', sep='\t')
    test_language = 'English'

    df_train['avg_token_freq'] = get_avg_token_freq(df_train['text'].tolist(), test_language)
    df_train['n_low_freq_words'] = get_n_low_freq_words(df_train['text'].tolist(), test_language)
    df_train['avg_word_length'] = get_avg_word_length_with_punct(df_train['text'].tolist())

    with open(args.out_path, 'w', encoding='utf8') as outfile:
        outfile.write('test_language,eyetracking-feature,model,mae,accuracy,r2,mse,rmse'+'\n')
    #
    #     for test_language in os.listdir(args.meco_path):
    #         if test_language != 'Estonian' and test_language != 'Italian':

        # read in MECO test data
        df_test = pd.read_csv(f'data/geco/english/test_scaled_and_correct_feats.tsv', sep='\t')

        df_test['avg_token_freq'] = get_avg_token_freq(df_test['text'].tolist(), test_language)
        df_test['n_low_freq_words'] = get_n_low_freq_words(df_test['text'].tolist(), test_language)
        df_test['avg_word_length'] = get_avg_word_length_with_punct(df_test['text'].tolist())

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
        "--test_set_type",
        type=str,
        default="matched",
        choices=["matched", "translated"],
        help="Specifies type of texts in the test set: either semantically matched or translated",
    )
    parser.add_argument(
        "--meco_path",
        type=str,
        default="data/meco/files_per_language",
        help="Path to the MECO files used for training and testing",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="results/baseline_results_geco_english_rbf_kernel.csv",
        help="Path to the file where the baseline results will be written",
    )

    args = parser.parse_args()
    compute_baselines_trained_on_geco(args)

    # # write out meco train combinations (crosslingual)
    # for test_language in os.listdir(args.meco_path):
    #     if test_language != 'Estonian' and test_language != 'Italian':
    #         df_train, df_test_matched, df_test_translated = combine_meco_data_crosslingual(args.meco_path, test_language)
    #         df_train.to_csv(f'{args.meco_path}/{test_language}/train.tsv', sep='\t', encoding='utf8', index=False, quoting=csv.QUOTE_NONE)
    #         df_test_matched.to_csv(f'{args.meco_path}/{test_language}/test_matched.tsv', sep='\t', encoding='utf8', index=False, quoting=csv.QUOTE_NONE)
    #         df_test_translated.to_csv(f'{args.meco_path}/{test_language}/test_translated.tsv', sep='\t', encoding='utf8', index=False, quoting=csv.QUOTE_NONE)

    # # write out meco train combinations (multilingual)
    # for trial_id in range(1, 13):
    #     df_train, df_test = combine_meco_data_multilingual(meco_path, trial_id)
    #
    #     if not os.path.exists(f'{meco_path_2}/{trial_id}'):
    #         os.makedirs(f'{meco_path_2}/{trial_id}')
    #
    #     df_train.to_csv(f'{meco_path_2}/{trial_id}/train.tsv', sep='\t', encoding='utf8', index=False, quoting=csv.QUOTE_NONE)
    #     df_test.to_csv(f'{meco_path_2}/{trial_id}/test.tsv', sep='\t', encoding='utf8', index=False, quoting=csv.QUOTE_NONE)

# def compute_baselines_trained_on_meco_multilingual(meco_path, out_path):
#
#     with open(out_path, 'w', encoding='utf8') as outfile:
#         outfile.write('test_trial_id,target,model,rmse,r2'+'\n')
#
#         for target_label in ['fix_count', 'first_pass_dur', 'tot_fix_dur', 'tot_regr_from_dur']:
#
#             # get train and test dataframes
#             for target_trial_id in range(1, 13):
#                 df_train, df_test = combine_meco_data_multilingual(meco_path, target_trial_id)
#
#                 # compute baselines
#                 for setting in ['linguistic', 'length', 'frequency', 'all']:
#                     results = svm(df_train, df_test, target_label, setting)
#                     outfile.write(f'{target_trial_id},{results["target"]},{results["train_features"]},{results["rmse"]},{results["r2"]}'+'\n')

# def combine_meco_data_multilingual(meco_path, target_trial_id):
#
#     all_data = pd.DataFrame()
#
#     for language in os.listdir(meco_path):
#
#         # We also exclude Estonian and Italian due to alignment errors
#         if language not in ['Estonian', 'Italian']:
#             df = pd.read_csv(f'{meco_path}/{language}/test.tsv', sep='\t', encoding='utf8', quoting=csv.QUOTE_NONE)
#
#             # add frequency columns to the dataframes
#             df['avg_token_freq'] = get_avg_token_freq(df['text'].tolist(), language)
#             df['n_low_freq_words'] = get_n_low_freq_words(df['text'].tolist(), language)
#
#             # concatenate language-specific dataframe to the full dataframe
#             all_data = pd.concat([all_data.reset_index(drop=True), df.reset_index(drop=True)])
#
#     # Instantiate train and test dataframe
#     df_train = pd.DataFrame()
#     df_test = pd.DataFrame()
#
#     for trial_id, df in all_data.groupby('trialid'):
#         # Only add the "unmatched" texts to the training data
#         if trial_id.isin([2,4,5,6,8,9,10]))
#             df_train = pd.concat([df_train.reset_index(drop=True), df.reset_index(drop=True)])
#
#     return df_train, df_test

# def combine_meco_data_crosslingual(meco_path, test_language):

#     # Compile the training dataframe: this includes the meco data of all languages, except the specified test language
#     df_train = pd.DataFrame()

#     for language in os.listdir(meco_path):

#         # We also exclude Estonian and Italian due to alignment errors
#         if language not in [test_language, 'Estonian', 'Italian']:
#             df = pd.read_csv(f'{meco_path}/{language}/test.tsv', sep='\t', encoding='utf8', quoting=csv.QUOTE_NONE)

#             # # add frequency columns to the dataframes
#             # df['avg_token_freq'] = get_avg_token_freq(df['text'].tolist(), language)
#             # df['n_low_freq_words'] = get_n_low_freq_words(df['text'].tolist(), language)

#             # concatenate language-specific dataframe to the full dataframe
#             df_train = pd.concat([df_train.reset_index(drop=True), df.reset_index(drop=True)])

#     # Read in the meco data of the test language
#     df_test = pd.read_csv(f'{meco_path}/{test_language}/test.tsv', sep='\t')

#     # # add frequency columns to the test dataframe
#     # df_test['avg_token_freq'] = get_avg_token_freq(df_test['text'].tolist(), test_language)
#     # df_test['n_low_freq_words'] = get_n_low_freq_words(df_test['text'].tolist(), test_language)

#     # Remove translated texts from the train df (otherwise the model sees the exact same texts during testing)
#     df_train = df_train[df_train['trialid'].isin([2,4,5,6,8,9,10])]

#     # Make two test sets: one with the semantically matched texts, and one with the translated texts
#     df_test_matched = df_test[df_test['trialid'].isin([2,4,5,6,8,9,10])]
#     df_test_translated = df_test[df_test['trialid'].isin([1,3,7,11,12])]

#     return df_train, df_test_matched, df_test_translated

# def compute_baselines_trained_on_meco_crosslingual(args):

#     all_languages = os.listdir(args.meco_path)

#     with open(args.out_path, 'w', encoding='utf8') as outfile:
#         outfile.write('test_language,test_set,eyetracking-feature,model,rmse,r2'+'\n')

#         for test_language in all_languages:
#             if test_language != 'Estonian' and test_language != 'Italian':
#                 for target_label in ['fix_count', 'first_pass_dur', 'tot_fix_dur', 'tot_regr_from_dur']:

#                     # get train and test dataframes
#                     df_train, df_test_matched, df_test_translated = combine_meco_data_crosslingual(args.meco_path, test_language)

#                     # compute baselines
#                     for setting in ['linguistic', 'length', 'frequency', 'all']:
#                         if args.test_set_type == 'matched':
#                             results = svm(df_train, df_test_matched, target_label, setting)
#                         elif args.test_set_type == 'translated':
#                             results = svm(df_train, df_test_translated, target_label, setting)
#                         outfile.write(f'{test_language},{args.test_set_type},{results["target"]},{results["train_features"]},{results["rmse"]},{results["r2"]}'+'\n')


