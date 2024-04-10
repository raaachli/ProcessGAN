from datetime import datetime

import pandas as pd


def process_timestamp(filename, savefile, savefile_2, caseID, start_time):
    df = pd.read_csv(filename, na_filter=False)
    caseids = df[caseID].values
    ucaseids = pd.unique(caseids)
    time_case_val = []
    case_dif_all = []
    with open(savefile, 'a') as f:
        for id in ucaseids:
            # print(id)
            df_case = df.loc[df[caseID] == id]
            df_case = df_case.reset_index(drop=True)

            if ':' in df_case[start_time]:
                df_case[start_time] = df_case[start_time].apply(time_to_seconds_2)
            else:
                df_case[start_time] = df_case[start_time].astype(float)
            df_case = df_case.sort_values(by=[start_time])
            df_case = df_case.reset_index(drop=True)
            time_case = []
            time_0 = df_case[start_time][0]
            time_end = df_case[start_time][len(df_case[caseID])-1]

            time_duration = int(float(time_end)-float(time_0))
            time_case_val.append(time_duration)
            for i in range(len(df_case[caseID])):

                time_0 = df_case[start_time][0]
                time_i = df_case[start_time][i]
                time_i = int(float(time_i)-float(time_0))

                time_case.append(int(time_i))
                if time_i < 0:

                    print(df_case['activity'][i])
                    print(id)
                    # print(df_case)

            max_time = max(time_case)


            time_case_dif = [0 for _ in range(len(time_case))]
            for i in range(1, len(time_case)):
                time_case_dif_i = time_case[i] - time_case[i-1]
                time_case_dif[i] = (time_case_dif_i/(max_time+0.000001))
            time_case_dif[0] = (0 / (max_time+0.000001))

            case_dif_all.append(time_case)


    min_val = min(time_case_val)
    max_val = max(time_case_val)
    mean_val = sum(time_case_val) / len(time_case_val)
    variance = sum((x - mean_val) ** 2 for x in time_case_val) / (len(time_case_val) - 1)
    std_val = variance ** 0.5
    for i in range(len(time_case_val)):
        time_case_val[i] = (time_case_val[i] - min_val) / (max_val - min_val)
    print(mean_val, std_val)



def time_to_seconds_2(t):
    # print(t)
    minutes, seconds = map(int, t.split(':'))
    return minutes * 60 + seconds

def time_to_seconds_4(t):
    # print(t)
    hours, minutes, seconds, _ = map(int, t.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def time_to_seconds_3(t):
    # print(t)
    hours, minutes, seconds = map(int, t.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def process_timestamp_sep(filename, savefile, savefile_2, caseID, start_time):
    df = pd.read_csv(filename, na_filter=False)
    caseids = df[caseID].values
    ucaseids = pd.unique(caseids)
    time_case_val = []
    case_dif_all = []
    with open(savefile, 'a') as f:
        for id in ucaseids:
            # print(id)

            df_case = df.loc[df[caseID] == id]
            df_case = df_case.reset_index(drop=True)
            # if len(df_case) > 50:
            time_case = []
            time_0 = df_case[start_time][0]
            time_end = df_case[start_time][len(df_case[caseID])-1]
            time_duration = compute_datetime_dif(time_0, time_end)
            # time_duration = int(float(time_end)-float(time_0))
            time_case_val.append(time_duration)

            for i in range(len(df_case[caseID])):

                time_0 = df_case[start_time][0]
                time_i = df_case[start_time][i]

                time_i = compute_datetime_dif(time_0, time_i)
                time_case.append(int(time_i))
                if time_i < 0:

                    print(df_case['Activity'][i])
                    print(id)
                    print(time_i)
                    # print(df_case)

            max_time = max(time_case)
            time_case_dif = [0 for _ in range(len(time_case))]
            for i in range(1, len(time_case)):
                time_case_dif_i = time_case[i] - time_case[i-1]
                time_case_dif[i] = (time_case_dif_i/(max_time+0.000001))
            time_case_dif[0] = (0 / (max_time+0.000001))

            case_dif_all.append(time_case)

    min_val = min(time_case_val)
    max_val = max(time_case_val)
    mean_val = sum(time_case_val)/len(time_case_val)
    variance = sum((x - mean_val) ** 2 for x in time_case_val) / (len(time_case_val) - 1)
    std_val = variance ** 0.5
    for i in range(len(time_case_val)):
        time_case_val[i] = (time_case_val[i]-min_val)/(max_val-min_val)
    print(mean_val, std_val)


def compute_datetime_dif(time_str1,time_str2):
    dt1 = datetime.strptime(time_str1, "%Y-%m-%d %H:%M:%S.%f")
    dt2 = datetime.strptime(time_str2, "%Y-%m-%d %H:%M:%S.%f")

    # Calculate time difference in hours
    delta = dt2 - dt1
    hours_difference = delta.days * 24 + delta.seconds / 3600
    return hours_difference


if __name__ == '__main__':
    data = 'SEP'
    file = 'data/data_raw/Sepsis Cases - Event Log.csv'
    save_file = 'data/data_time/'+data+'/data_seq/'+data+'_time_dif_norm.txt'
    save_file2 = 'data/data_time/'+data+'/data_seq/'+data+'_time_duration_norm.txt'
    caseID, start_time = 'Case ID', 'Complete Timestamp'
    process_timestamp_sep(file, save_file, save_file2, caseID, start_time)
    # caseID, start_time = 'caseID', 'startTime'
    # process_timestamp(file, save_file, save_file2, caseID, start_time)
