import pm4py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import gc
import time
import psutil
import os
import polars as pl
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.telegram import tqdm

file_names = ["Hospital_log"]
attribute_keys = ["Age:1"]

max_worker = psutil.cpu_count() - 4

################################

TG_BOT_TOKEN = "6746267594:AAFMB3DKeyPjyzlcAD5KJRUAs4UmwzOsrPU"
CHAT_ID = "2019745574"

def tg_send_msg(msg):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={msg}"
    print(requests.get(url).json())

def tg_send_file(path):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendDocument?"
    data = {
        'chat_id': CHAT_ID,
        'parse_mode':'HTML',
    }
    files = {
        'document': open(path, 'rb')
    }
    print(requests.post(url, data=data, files=files, stream=True).json())

tg_send_msg("connection established")   

################################

if __name__ == '__main__':

    for file_name, attribute_key in zip(file_names, attribute_keys):

        case_attribute_key = f'case:{attribute_key}'
        case_id_key = "case:concept:name"

        print(f"reading ../logs/{file_name}.xes")

        # df = pm4py.read_xes(f'../logs/{file_name}.xes')
        df = pl.read_csv("../logs/{file_name}.{attribute_key}.light.csv")

        df = df.dropna(subset=[case_attribute_key]).reset_index() # filter empty rows 

        attribute_values = pm4py.get_trace_attribute_values(df, attribute_key)
        sorted_attribute_values = sorted([ float(k) for k in attribute_values.keys() ])

################################

        variants = pm4py.get_variants_as_tuples(df)
        activities = {}

        for (v, o) in variants.items():
            for a in v:
                activities[a] = activities.setdefault(a, 0) + o

        activities = dict(sorted(activities.items(), key=lambda x: x[1], reverse=True))
        activity_to_char = { k: chr(i) for i, (k, v) in enumerate(activities.items()) }
        print(len(activity_to_char))

        def trace_to_string(t):
            return "".join([ activity_to_char[a] for a in t ])
        
        from rapidfuzz.distance import Levenshtein

        def LD(A, B):
            return Levenshtein.distance(A, B)

        def NGLD(A, B):
            ld = LD(A, B)
            return (2 * ld) / ((len(A) + len(B)) + ld)
        
        def LNGLD(L1, L2, S1, S2):
            L1_string_log = { trace_to_string(k): v for k, v in L1.items() }
            L2_string_log = { trace_to_string(k): v for k, v in L2.items() if v > 0 }

            sum = 0
            for t1, n1 in L1_string_log.items():
                for t2, n2 in L2_string_log.items():
                    sum += NGLD(t1, t2) * n1 * n2

            return sum / (S1 * S2)
        
        def MMLNGLD(L1, L2, S1, S2):
            return ((1 - LNGLD(L1, L2, S1, S2)) + LNGLD(L1, L1, S1, S1) + LNGLD(L2, L2, S2, S2)) / 3
        
################################

        # group df by case ids
        grouped = df.filter([case_id_key, case_attribute_key]).groupby(case_id_key, sort=False)

        memory = psutil.virtual_memory().available
        file_size = os.path.getsize(f'../logs/{file_name}.xes')
        num_worker = int(min(max_worker, (memory) //  (3 * file_size)))

        print(f'num worker: min({max_worker}, {memory} // {3 * file_size}) = {num_worker}')

        segments = np.array_split(sorted_attribute_values[1:], num_worker)

        tg_send_msg(f"{file_name}, {attribute_key}: {num_worker} workers, each {len(segments[0])} elements")

        def compute(input):
            segment, id = input

            df_gte_light = grouped.first() # initial gte group

            # define absolute languages (counting the occurrence of cases)
            # devide absoulte numbers by total number of cases later(!), otherwise every values has to be updated
            language_lt = {}
            language_gte = pm4py.stats.get_variants(df)
       
            # # s1s = []
            # # s2s = []
            ds = []

            for pivot in tqdm(segment, position=id, desc=f"worker {id: >2}", token=TG_BOT_TOKEN, chat_id=CHAT_ID, mininterval=1):

                # find affected cases
                df_affected_light_idx = df_gte_light[df_gte_light[case_attribute_key] < pivot].index

                # remove affected cases from gte group
                df_gte_light.drop(df_affected_light_idx, inplace=True)

                # translate into full df
                df_affected = pd.concat([ df.loc[grouped.indices[i]] for i in df_affected_light_idx ])

                # update absoulte languages
                for tr, n in pm4py.stats.get_variants(df_affected).items():
                    language_lt[tr] = language_lt.setdefault(tr, 0) + n
                    language_gte[tr] = language_gte[tr] - n

                language_lt_size = sum(language_lt.values(), 0.0)
                language_gte_size = sum(language_gte.values(), 0.0)

                # # s1 = LNGLD(language_lt, language_lt, language_lt_size, language_lt_size)
                # # s2 = LNGLD(language_gte, language_gte, language_gte_size, language_gte_size)
                d = LNGLD(language_lt, language_gte, language_lt_size, language_gte_size)
                
                # collect data to plot graph
                # # s1s.append(s1)
                # # s2s.append(s2)
                ds.append(d)
            
            return ds # # list(zip(s1s, s2s, ds))

        matrix = process_map(compute, zip(segments, list(range(1, num_worker+1))), max_workers=num_worker)

        results = [
            r
            for col in matrix
            for r in col
        ]

        time.sleep(2)

        # # (s1s, s2s, ds) = zip(*results)

        # # mmlnglds = [ (d + (2 - s1 - s2) / 2) / 2 for d, s1, s2 in zip(ds, s1s, s2s) ]

        # # pd.DataFrame({"d": ds, "s1": s1s, "s2": s2s, "mmlngld": mmlnglds}, copy=False).to_csv(f"out.mmlngld.{file_name}.{attribute_key}.mt.csv", index=False, header=False)
        # # tg_send_file(f"out.mmlngld.{file_name}.{attribute_key}.mt.csv")

        pd.DataFrame({"lngld": results}, copy=False).to_csv(f"out.lngld.{file_name}.{attribute_key}.mt.csv", index=False, header=False)
        tg_send_file(f"out.lngld.{file_name}.{attribute_key}.mt.csv")

################################

        plt.figure(figsize=(10,6))

        # plot emd
        plt.plot(sorted_attribute_values[1:], results, label=r"$\Delta(L_1,L_2)$")
        plt.legend()
        plt.savefig(f"out.lngld.{file_name}.{attribute_key}.mt.png", bbox_inches="tight")
        tg_send_file(f"out.lngld.{file_name}.{attribute_key}.mt.png")

################################

        del df
        gc.collect()


