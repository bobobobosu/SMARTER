"""
To get a "time-related" subset of WikiTableQuestions v1.0.2
* I've removed the unrelated files, keeping only the relevant dirs (csv/ and tagged/)
* Only csv/*.csv are processed since they are referred to first.

Define "time-related":

'tagged/':
    df.columns:
        ['row', 'col', 'id', 'content', 'tokens', 'lemmaTokens', 'posTags',
        'nerTags', 'nerValues', 'number', 'date', 'num2', 'list', 'listId']
        => 'date': Let's use this!
    'nerTags' includes:
        {'ORDINAL', 'DURATION', 'MISC', 'SET', 'O', 'NUMBER', 'nan', 'MONEY', 
        'TIME', 'PERSON', 'DATE', 'ORGANIZATION', 'PERCENT', 'LOCATION'}
        => 'DATE', 'TIME'
"""
import os
import pandas as pd


def get_temporal_subset(WikiTableQuestionsPath, ratio_threshold=0.1):
    list_of_tables = []
    # out_dir = "csv-temporal-0.1"
    # ratio_threshold = (
    #     0.1  # 0.1 => subset of 47%; 0.2 => subset of 9% (See ``stats.txt```)
    # )

    # def mkdir_if_not_dir(dir_path):
    #     if not os.path.isdir(dir_path):
    #         os.mkdir(dir_path)

    ctr1, ctr2, ctr3 = 0, 0, 0
    # mkdir_if_not_dir(out_dir)
    nerTags = set()
    for root, dirs, files in os.walk(WikiTableQuestionsPath):
        if not root.endswith("-csv"):
            continue
        tagged_root = root.replace("csv", "tagged")
        segs = root.split(os.path.sep)
        # out_root = os.path.sep.join([*segs[:-2], out_dir, (segs[-1] + "-temporal")])
        # mkdir_if_not_dir(out_root)

        for name in files:
            if not name.endswith(".csv"):
                continue
            ctr1 += 1
            try:
                with open(os.path.join(root, name), "r") as csv_f:
                    csv_df = pd.read_csv(csv_f)
                with open(
                    os.path.join(tagged_root, name.replace("csv", "tagged")), "r"
                ) as tagged_f:
                    tagged_df = pd.read_csv(tagged_f, sep="\t")

                tagged_df_drop_nas = tagged_df.dropna(subset=["date"])
                tagged_df_temp = tagged_df.loc[tagged_df['nerTags'].isin(['DATE','TIME','DURATION'])]
                temporol_columns = csv_df.iloc[:,list(set(tagged_df_temp['col']))]

                if len(tagged_df_drop_nas) > int(len(tagged_df) * ratio_threshold):
                    date_columns = None
                    list_of_tables += [temporol_columns]
                    # csv_df.to_csv(os.path.join(out_root, name))
                    ctr2 += 1
            except Exception as e:
                print("Failed to process {}: {}".format(name, e))
                ctr3 += 1

    stats = "\n".join(
        [
            "Threshold = {}".format(ratio_threshold),
            "A subset of {} over {} ({:.1f}%) created.".format(
                ctr2, ctr1, ctr2 / ctr1 * 100
            ),
            "{} failed ({:.1f}%).".format(ctr3, ctr3 / ctr1 * 100),
        ]
    )
    print(stats)
    # with open(os.path.join(out_dir, "stats.txt"), "w") as stats_f:
    #     stats_f.write(stats + "\n")

    return list_of_tables
