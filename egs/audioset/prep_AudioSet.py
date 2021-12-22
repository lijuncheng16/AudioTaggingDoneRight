
import numpy as np
import json
import os
import zipfile
import wget
import csv
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--slurm-id", type=str, default='5689254', help="the root path of audio data")

if __name__ == '__main__':
    args = parser.parse_args()
    base_path = '/local/slurm-' + args.slurm_id + '/local/audio'
    list_path = "/jet/home/billyli/data_folder/DayLongAudio/lists/"

    # fix bug: generate an empty directory to save json files
    if os.path.exists(base_path + '/data/datafiles') == False:
        os.mkdir(base_path + '/data/datafiles')

    bal_train_wav_list = []
    unbal_train_wav_list = []
    eval_wav_list = []
    bal_unbal_train_wav_list = []
    with open(os.path.join(list_path, 'unbalanced_train_segments.csv')) as unbal_csv_file:
        unbal_csv_reader = csv.reader(unbal_csv_file, delimiter=',')

        unbal_count = 0
        missing_count = 0
        for row in unbal_csv_reader:
            if unbal_count <3:
                print(f'Column names are {", ".join(row)}')
                unbal_count += 1
            else:
                wav_path = os.path.join(base_path, 'unbalanced_wav',row[0] +'.wav')
                if os.path.exists(wav_path):
                    cur_unbal_dict = {"wav": wav_path, "labels": eval(','.join(row[3:]))}
    #             print(cur_unbal_dict)
                    unbal_train_wav_list.append(cur_unbal_dict)
                    bal_unbal_train_wav_list.append(cur_unbal_dict)
                    unbal_count += 1

                else:
                    missing_count+=1
    #                 print("missing training: " + row[0])
    print(f'unbalanced count: {unbal_count}, missing count: {missing_count}')
    with open(base_path + '/data/datafiles/audioset_unbal_train_data' +'.json', 'w') as f:
        json.dump({'data': unbal_train_wav_list}, f, indent=1)

    with open(os.path.join(list_path, 'balanced_train_segments.csv')) as bal_csv_file:
        bal_csv_reader = csv.reader(bal_csv_file, delimiter=',')

        bal_count = 0
        missing_count = 0
        for row in bal_csv_reader:
            if bal_count <3:
                print(f'Column names are {", ".join(row)}')
                bal_count += 1
            else:
                wav_path = os.path.join(base_path, 'balance_wav',row[0] +'.wav')
                if os.path.exists(wav_path):
                    cur_bal_dict = {"wav": wav_path, "labels": eval(','.join(row[3:]))}
    #             print(cur_bal_dict)
                    bal_train_wav_list.append(cur_bal_dict)
                    bal_unbal_train_wav_list.append(cur_bal_dict)
                    bal_count += 1

                else:
                    missing_count+=1
    #                 print("missing training: " + row[0])
    print(f'balanced count: {bal_count}, missing count: {missing_count}')
    with open(base_path + '/data/datafiles/audioset_bal_train_data' +'.json', 'w') as f:
        json.dump({'data': bal_train_wav_list}, f, indent=1)

    with open(base_path + '/data/datafiles/audioset_bal_unbal_train_data' +'.json', 'w') as f:
        json.dump({'data': bal_unbal_train_wav_list}, f, indent=1)

    with open(os.path.join(list_path, 'eval_segments.csv')) as eval_csv_file:
        eval_csv_reader = csv.reader(eval_csv_file, delimiter=',')

        eval_count = 0
        missing_count = 0
        for row in eval_csv_reader:
            if eval_count <3:
                print(f'Column names are {", ".join(row)}')
                eval_count += 1
            else:
                wav_path = os.path.join(base_path, 'eval_wav',row[0] +'.wav')
                if os.path.exists(wav_path):
                    cur_eval_dict = {"wav": wav_path, "labels": eval(','.join(row[3:]))}
    #             print(cur_bal_dict)
                    eval_wav_list.append(cur_eval_dict)
                    eval_count += 1

                else:
                    missing_count+=1
    #                 print("missing training: " + row[0])
    print(f'eval count: {eval_count}, missing count: {missing_count}')
    with open(base_path + '/data/datafiles/audioset_eval_data' +'.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)


    print('Finished AudioSet Preparation')
