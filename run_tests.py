from main import simulate
from settings.test_parameters import Parameters
import utils
from datetime import datetime
import os
import pandas as pd
from dataclasses import asdict
from tqdm import tqdm

def convert_dict_values_to_str(dictionary):
    res = dict()
    delim = " | "
    for sub in dictionary:
        if isinstance(dictionary[sub], list) or isinstance(dictionary[sub], tuple):
            res[sub] = delim.join([str(ele) for ele in dictionary[sub]])
        else:
            res[sub] = str(dictionary[sub])

    return res
def run_test(test_result_directory="test_results"):
    if not os.path.exists(test_result_directory):
        os.makedirs(test_result_directory)

    test_parameters = [
        Parameters(sigma_a=9, i_list=(12,), antenna_kind="2_6-3-8"),
        Parameters(sigma_a=9, i_list=(12, 14,), antenna_kind="2_6-3-8"),
        Parameters(sigma_a=9, i_list=(12, 14, 16,), antenna_kind="2_6-3-8"),
    ]

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    df = pd.DataFrame()

    for i, parameters in enumerate(test_parameters):
        print(f"\n\nPerforming test {i}")
        print("Parameter configuration:")
        print(parameters)
        xs, beacon_pos, antenna_list = simulate(params=parameters)
        rmse = utils.rmse(xs[:, :3].T, beacon_pos)
        parameters_dict = asdict(parameters)
        parameters_dict_str = convert_dict_values_to_str(parameters_dict)

        cur = pd.DataFrame(parameters_dict_str, index=[i])
        cur["rmse"] = rmse
        df = pd.concat([df, cur], ignore_index=True)


    filename = date_time + ".csv"
    filepath = os.path.join(test_result_directory, filename)
    print(filepath)
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    run_test()
