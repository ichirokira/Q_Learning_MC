import numpy as np
from scipy.spatial import distance
import csv
from Node_Method import find_receiver
from Q_learning_method import init_function, action_function, q_max_function, reward_function
from regression import Regression
import Parameter as para

class Q_learning:
    def __init__(self, index, writer_w=None, information_log_w=None, init_func=init_function, nb_action=81, action_func=action_function, network=None):
        self.action_list = action_func(nb_action=nb_action)  # the list of action
        self.q_table = init_func(nb_action=nb_action)  # q table
        self.state = nb_action  # the current state of actor
        self.charging_time = [0.0 for _ in self.action_list]  # the list of charging time at each action
        self.reward = np.asarray([0.0 for _ in self.action_list])  # the reward of each action
        self.reward_max = [0.0 for _ in self.action_list]  # the maximum reward of each action
        self.reg = Regression(startAt=0)
        self.W = np.array([0.5, 0.5])
        self.reg_number = 0  # number of data saved in file log
        self.index = index
        self.writer_w = writer_w
        self.information_log_w = information_log_w
        print(self.writer_w)


    def update(self, network, mc_current_location=None,alpha=0.5, gamma=0.6, q_max_func=q_max_function, reward_func=reward_function):
        if not len(network.mc.list_request):
            return self.action_list[self.state], 0.0, 0.0, 0.0
        first, second =  self.set_reward(reward_func=reward_func, network=network, location=mc_current_location)
        self.q_table[self.state] = (1 - alpha) * self.q_table[self.state] + alpha * (
                self.reward + gamma * self.q_max(q_max_func))
        self.choose_next_state(network)
        if self.state == len(self.action_list) - 1:
            charging_time = (network.mc.capacity - network.mc.energy) / network.mc.e_self_charge
        else:
            charging_time = self.charging_time[self.state]
        print("next state =", self.action_list[self.state], self.state, charging_time)
        print(self.charging_time)
        return self.action_list[self.state], charging_time, first[self.state], second[self.state]

    def q_max(self, q_max_func=q_max_function):
        return q_max_func(q_table=self.q_table, state=self.state)

    def set_reward(self, reward_func=reward_function, network=None, location=None):

        first = np.asarray([0.0 for _ in self.action_list], dtype=float)
        second = np.asarray([0.0 for _ in self.action_list], dtype=float)

        # third = np.asarray([0.0 for _ in self.action_list], dtype=float)
        for index, row in enumerate(self.q_table):
            temp = reward_func(network=network, q_learning=self, state=index, receive_func=find_receiver)
            first[index] = temp[0]
            second[index] = temp[1]
            # third[index] = temp[2]
            self.charging_time[index] = temp[2]
        first = first / (np.sum(first) + 1e-8)
        second = second / (np.sum(second) + 1e-8)
        print("[INFO] First, Second", first, second)
        # third = third / np.sum(third)


        with open("./log/Q_learning2nd/thaydoisonode/result_10_intervals/gamma0.6/regression_data"+str(self.index)+".csv", 'r') as csvfile:
            csv_dict = [row for row in csv.DictReader(csvfile)]
            print("[INFO] Length file", len(csv_dict))
            if len(csv_dict) != 0:

                self.reg.read_data(train_filename="./log/Q_learning2nd/thaydoisonode/result_10_intervals/gamma0.6/regression_data"+str(self.index)+".csv", target_filename="./log/Q_learning2nd/thaydoisonode/result_10_intervals/gamma0.6/regression_target_data"+str(self.index)+".csv")
                print("[INFO] Length truth: ", len(self.reg.delta))
                print("[INFO] Para X", para.X)
        print("[INFO] location", location)
        if para.X != 0:
            if((len(self.reg.delta)-1) % para.X == 0) and len(self.reg.delta) != 1 and location !=para.depot:
                print("[INFO] Update")
                print("[INFO] StartAT: ", self.reg.startAt)
                self.W = self.reg.update()
                print("[INFO] StartAT: ", self.reg.startAt)
                self.W = self.W / (np.sum(self.W) + 1e-8)
                print("Parameters: ", self.W)
                self.writer_w.writerow({"Weights": self.W})
                self.information_log_w.flush()
        self.reward = self.W[0] * first + self.W[1] * second
        self.reward_max = list(zip(first, second))
        return first, second

    def choose_next_state(self, network):
        # next_state = np.argmax(self.q_table[self.state])
        if network.mc.energy < 10:
            self.state = len(self.q_table) - 1
        else:
            self.state = np.argmax(self.q_table[self.state])
            #print(self.reward_max[self.state])
            #print(self.action_list[self.state])


