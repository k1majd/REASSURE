import sys, os
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch, torchvision, time, math, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from REASSURE.ExperimentModels import MLP
from REASSURE.Repair import REASSURERepair
from example.nnet_reader import HCAS_Model
from example.h5_reader import My_H5Dataset
from REASSURE.ExperimentTools import constraints_from_labels
import tensorflow as tf
import pickle
import argparse
from csv import writer


def arg_parser():
    """_summary_

    Returns:
        _type_: _description_
    """
    # cwd = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-id",
        "--modelIndex",
        nargs="?",
        type=int,
        default=0,
        help="index of repair model.",
    )
    return parser.parse_args()


def success_rate(model, buggy_inputs, right_label, is_print=0):
    with torch.no_grad():
        pred = model(buggy_inputs)
        correct = (pred.argmax(1) == right_label).type(torch.float).sum().item()
    if is_print == 1:
        print(
            "Original accuracy on buggy_inputs: {} %".format(
                100 * correct / len(buggy_inputs)
            )
        )
    elif is_print == 2:
        print("Success repair rate: {} %".format(100 * correct / len(buggy_inputs)))
    return correct / len(buggy_inputs)


def original_data_loader():
    load_str = "_5_31_2022_16_35_50"
    if not os.path.exists(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_ex/global_constraint/data"
    ):
        os.makedirs(
            os.path.dirname(os.path.realpath(__file__))
            + "/repair_ex/global_constraint/data"
        )
    with open(
        os.path.dirname(os.path.realpath(__file__))
        + f"/repair_ex/global_constraint/data/repair_dataset{load_str}.pickle",
        "rb",
    ) as data:
        dataset = pickle.load(data)
    return dataset[0], dataset[1]
    # return dataset[0], dataset[1], dataset[2], dataset[3]


class neural_network(nn.Module):
    def __init__(self, input_size, output_size):
        super(neural_network, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=32, out_features=32)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=32, out_features=output_size)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)

        return x


def get_params(model_torch, weights):
    for m_perb, d in zip(model_torch.parameters(), weights):
        m_perb.data = torch.tensor(d).T
    return model_torch


def contstraint_bounding(num_samp):
    A = []
    b = []
    for i in range(num_samp):
        A.append(np.array([[1]]))
        b.append(np.array([9.5]))
    return A, b


def loadData(name_csv):
    with open(name_csv) as csv_file:
        data = np.asarray(list(csv.reader(csv_file, delimiter=",")), dtype=np.float32)
    return data


def generateDataWindow(window_size):
    Dfem = loadData(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_ex/global_constraint/data/GeoffFTF_1.csv"
    )
    Dtib = loadData(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_ex/global_constraint/data/GeoffFTF_2.csv"
    )
    Dfut = loadData(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_ex/global_constraint/data/GeoffFTF_3.csv"
    )
    n = 20364
    Dankle = np.subtract(Dtib[:n, 1], Dfut[:n, 1])
    observations = np.concatenate((Dfem[:n, 1:], Dtib[:n, 1:]), axis=1)
    observations = (observations - observations.mean(0)) / observations.std(0)
    controls = Dankle  # (Dankle - Dankle.mean(0))/Dankle.std(0)
    n_train = 18200
    # n_train = 500
    train_observation = np.array([]).reshape(0, 4 * window_size)
    test_observation = np.array([]).reshape(0, 4 * window_size)
    for i in range(n_train):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        train_observation = np.concatenate((train_observation, temp_obs), axis=0)
    train_controls = controls[window_size : n_train + window_size].reshape(-1, 1)
    for i in range(n_train, n - window_size):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        test_observation = np.concatenate((test_observation, temp_obs), axis=0)
    test_controls = controls[n_train + window_size :].reshape(-1, 1)
    return train_observation, train_controls, test_observation, test_controls


def give_stats(model, x, y, y_pred_orig, bound):
    y_pred_new = model(x)
    x = x.detach().numpy()
    y = y.detach().numpy()
    y_pred_orig = y_pred_orig.detach().numpy()
    y_pred_new = y_pred_new.detach().numpy()

    # find violations
    x_temp = []
    y_temp = []
    for i in range(x.shape[0]):
        if y_pred_orig[i][0] > bound:
            x_temp.append(x[i])
            y_temp.append(y_pred_new[i])

    num_violations = len(x_temp) * 1.0
    num_no_violations = num_violations
    for i in range(len(x_temp)):
        if y_temp[i][0] > bound:
            num_no_violations -= 1.0
    satisfaction_rate = num_no_violations / num_violations

    # find mae
    x_temp = []
    y_orig = []
    y_new = []
    for i in range(x.shape[0]):
        if y_pred_orig[i][0] <= bound:
            x_temp.append(x[i])
            y_orig.append(y[i])
            y_new.append(y_pred_new[i])
    x_temp = np.array(x_temp)
    y_orig = np.array(y_orig)
    y_new = np.array(y_new)
    mae = np.mean(np.abs(y_orig - y_new))

    # introduced bugs
    idx_violation = np.where(y_pred_new > bound)[0]
    idx_orig_no_violate = np.where(y_pred_orig[idx_violation] < bound)[0]
    if len(idx_violation) != 0:
        intro_bug = len(idx_orig_no_violate) / len(idx_violation)
    else:
        intro_bug = 0

    return satisfaction_rate, mae, intro_bug


def Repair_HCAS(repair_num, n, id):
    train_obs, train_ctrls, test_obs, test_ctrls = generateDataWindow(10)
    for id in range(50):
        # load the original model
        model_orig = tf.keras.models.load_model(
            os.path.dirname(os.path.realpath(__file__))
            + f"/repair_ex/global_constraint/model_orig/model_orig_{id}"
        )
        weights = model_orig.get_weights()
        # x_train, y_train, x_test, y_test = original_data_loader()

        # data generation
        x_train, y_train = original_data_loader()
        inp_max = []
        inp_min = []
        # for i in range(x_train.shape[1]):
        #     inp_max.append(np.maximum(x_train[:, i].max(), x_test[:, i].max()))
        #     inp_min.append(np.minimum(x_train[:, i].min(), x_test[:, i].min()))
        for i in range(x_train.shape[1]):
            inp_max.append(x_train[:, i].max())
            inp_min.append(x_train[:, i].min())
        inp_bound = [np.array(inp_max), -np.array(inp_min)]
        adv_idx = np.where(model_orig.predict(x_train) > 10)[0]
        x_train = x_train[adv_idx]
        y_train = y_train[adv_idx]
        x_train = torch.tensor(x_train).float()
        y_train = torch.tensor(y_train).float()
        # x_test = torch.tensor(x_test).float()
        # y_test = torch.tensor(y_test).float()
        model_torch = MLP(size=[40, 32, 32, 32, 1])
        model_torch = get_params(model_torch, weights)
        input_dim = 40
        input_boundary = [
            np.block([[np.eye(input_dim)], [-np.eye(input_dim)]]),
            np.block(inp_bound),
        ]

        output_constraints = contstraint_bounding(x_train[:repair_num].shape[0])
        x_train = x_train[:repair_num]
        # success_rate(model_torch, buggy_inputs, right_labels, is_print=1)
        start = time.time()
        REASSURE = REASSURERepair(model_torch, input_boundary, n)
        repaired_model = REASSURE.point_wise_repair(
            x_train, output_constraints=output_constraints, core_num=1
        )
        # x_test = torch.tensor(test_obs).float()
        # y_test = torch.tensor(test_ctrls).float()
        # y_pred = repaired_model(x_test)
        # plt.plot(y_test.detach().numpy(), label="true")
        # plt.plot(y_pred.detach().numpy(), label="pred")
        # plt.legend()
        # plt.show()
        cost_time = time.time() - start
        # print("Time:", cost_time)
        # error = y_test.detach().numpy() - y_pred.detach().numpy()
        # pickle.dump(
        #     [y_pred.detach().numpy().flatten(), error.flatten()],
        #     open("model_torch_bound_data.pkl", "wb"),
        # )

        sat_rate, _, intro_bug = give_stats(
            repaired_model,
            torch.tensor(test_obs).float(),
            torch.tensor(test_ctrls).float(),
            torch.tensor(model_orig.predict(test_obs)).float(),
            10,
        )
        _, mae, _ = give_stats(
            repaired_model,
            torch.tensor(test_obs).float(),
            torch.tensor(test_ctrls).float(),
            torch.tensor(model_orig.predict(test_obs)).float(),
            9.5,
        )
        print("satisfaction rate:", sat_rate)
        print("mae:", mae)
        print("cost time:", cost_time)
        print("introduced bugs:", intro_bug)

        with open(
            os.path.dirname(os.path.realpath(__file__))
            + f"/repair_ex/global_constraint/stats/bound_stat.csv",
            "a+",
            newline="",
        ) as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            model_evaluation = [
                "model",
                id,
                "satisfaction rate",
                sat_rate,
                "mae",
                mae,
                "cost time",
                cost_time,
                "introduced bugs",
                intro_bug,
            ]
            # Add contents of list as last row in the csv file
            csv_writer.writerow(model_evaluation)
    # success_rate(repaired_model, buggy_inputs, right_labels, is_print=2)


if __name__ == "__main__":
    args = arg_parser()
    model_id = args.modelIndex
    for num in [25]:
        print("-" * 50, "number:", num, "-" * 50)
        Repair_HCAS(num, 1, model_id)
