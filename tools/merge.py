import os, fnmatch
import pandas as pd

def merge_csv_to_df(path, pattern):
    current_dir = os.getcwd()
    os.chdir(path)
    current_dir = os.getcwd()
    listOfFilesToRemove = os.listdir('./')
    #pattern = "*.csv"
    li = []
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("csv file : ",entry)
            df = pd.read_csv(entry, index_col=None, header=0)
            li.append(df)
            os.remove(entry)

    df_frame = pd.concat(li, axis=0, ignore_index=True)

    # today = date.today()
    # df_frame.to_csv("stocks_movments_merged_" + str(today) + ".csv", index=True, sep='\t')

    os.chdir(current_dir)

    return df_frame


def merge_list(input_files):
    wipe_out_directory(config.OUTPUT_DIR_MERGED)

    list_files = get_input_list(input_files)
    # print(list_files)

    file_merge_list = []
    for dir in config.OUTPUT_LIST_DIR:
        for f in os.listdir(dir):
            if (f in list_files):
                file_merge_list.append(os.path.join(dir, f))
                shutil.copyfile(os.path.join(dir, f), os.path.join(config.OUTPUT_DIR_MERGED, f))
    # print(file_merge_list)

    df_merged = merge_csv_to_df(config.OUTPUT_DIR_MERGED, '*.csv')
    df_merged = drop_df_duplicates(df_merged, "symbol")

    filename = config.OUTPUT_DIR_RESULT + 'symbol_list_' + input_files
    df_merged.to_csv(filename)
