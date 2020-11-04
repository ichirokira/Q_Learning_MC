from scipy.spatial import distance

import Parameter as para
from MobileCharger_Method import get_location, charging
import csv
import numpy as np

class MobileCharger:
    def __init__(self, index, writer_t=None, writer_d=None, information_log_t = None, information_log_d=None, energy=None, e_move=None, start=para.depot, end=para.depot, velocity=None,
                 e_self_charge=None, capacity=None):
        self.is_stand = False  # is true if mc stand and charge
        self.is_self_charge = False  # is true if mc is charged
        self.is_active = False  # is false if none of node request and mc is standing at depot

        self.start = start  # from location
        self.end = end  # to location
        self.current = start  # location now
        self.end_time = -1  # the time when mc finish charging

        self.energy = energy  # energy now
        self.capacity = capacity  # capacity of mc
        self.e_move = e_move  # energy for moving
        self.e_self_charge = e_self_charge  # energy receive per second
        self.velocity = velocity  # velocity of mc

        self.list_request = []  # the list of request node
        self.index = index
        self.writer_t = writer_t
        self.writer_d = writer_d
        self.temp1 = 0
        self.energy1 = 0
        self.dead1 = 0
        self.avg_en1 = 0
        self.temp2 = 0

        self.avg_en2 = 0
        self.first = 0
        self.second = 0
        self.information_log_t = information_log_t
        self.information_log_d = information_log_d
        print(self.writer_t)
        print(self.writer_d)
    def update_location(self, func=get_location):
        self.current = func(self)
        self.energy -= self.e_move

    def charge(self, network=None, node=None, charging_func=charging):
        charging_func(self, network, node)

    def self_charge(self):
        self.energy = min(self.energy + self.e_self_charge, self.capacity)

    def check_state(self):
        if distance.euclidean(self.current, self.end) < 1:
            self.is_stand = True
            self.current = self.end
        else:
            self.is_stand = False
        if distance.euclidean(para.depot, self.end) < 10 ** -3:
            self.is_self_charge = True
        else:
            self.is_self_charge = False

    def get_next_location(self, network, time_stem, optimizer=None):
        next_location, charging_time, first, second = optimizer.update(network, mc_current_location=self.current)
        self.start = self.current
        self.end = next_location
        moving_time = distance.euclidean(self.start, self.end) / self.velocity
        self.end_time = time_stem + moving_time + charging_time
        self.end_time = int(self.end_time) + 1
        return first, second

    def run(self, network, time_stem, optimizer=None):
        # print(self.energy, self.start, self.end, self.current)


        if (not self.is_active and self.list_request) or abs(
                time_stem - self.end_time) < 1:
            self.is_active = True
            self.list_request = [request for request in self.list_request if
                                 network.node[request["id"]].energy < network.node[request["id"]].energy_thresh]
            if not self.list_request:
                self.is_active = False

            self.first, self.second = self.get_next_location(network=network, time_stem=time_stem, optimizer=optimizer)
            self.energy1 = [network.node[i].energy for i in range(len(network.node))]
            self.dead1 = [i for i in range(len(network.node)) if network.node[i].is_active is False]
            self.temp1 = time_stem
            self.avg_en1 = [network.node[i].avg_energy for i in range(len(network.node))]
            min_index = np.argmin([network.node[i].energy / (network.node[i].avg_energy+1e-8) for i in range(len(network.node)) if network.node[i].is_active is True])
            #print("[INFO] avg_en1: {}, temp1: {}, min_energy: {}, min_avg_energy: {}".format( self.avg_en1, self.temp1, network.node[min_index].energy,
                                                                                #network.node[min_index].avg_energy))
            check = self.end_time

        else:
            if self.is_active:

                if not self.is_stand:
                    #print("moving")
                    self.update_location()
                elif not self.is_self_charge:
                    #print("charging")


                    self.charge(network)
                    #print("[INFO] Printing charging end time", self.end_time)
                    #print("[INFO] Printing current time", time_stem)
                    if (time_stem == (self.end_time - 2)) :
                        self.temp2 = self.end_time
                        self.avg_en2 = [network.node[i].avg_energy for i in range(len(network.node))]
                        min_index = np.argmin([network.node[i].energy for i in range(len(network.node)) if
                                               network.node[i].is_active is True])
                        print("[INFO] avg_en2: {}, temp2: {}, min_energy: {}, min_avg_energy: {}".format(self.avg_en2, self.temp2, network.node[min_index].energy, network.node[min_index].avg_energy))

                        avg = [(self.avg_en2[i] + self.avg_en1[i])/2 for i in range(len(self.avg_en1))]
                        self.temp1 += min([(self.energy1[i] / (avg[i]+1e-8)) for i in range(len(network.node)) if i not in self.dead1])
                        self.temp2 += + min([(network.node[i].energy / (avg[i]+1e-8)) for i in range(len(network.node)) if network.node[i].is_active is True])
                        delta = self.temp2 - self.temp1
                        print("[INFO] Print time log: {}, {}".format(self.temp2, self.temp1))

                        self.writer_t.writerow({"delta": delta})
                        self.information_log_t.flush()
                        print("[INFO] Print log Elements: {}, {}".format(self.first, self.second))
                        self.writer_d.writerow(({"E_ele": self.first, "M_ele": self.second}))
                        self.information_log_d.flush()
                else:
                    #print("self charging")
                    self.self_charge()
        if self.energy < para.E_mc_thresh and not self.is_self_charge and self.end != para.depot:
            self.start = self.current
            self.end = para.depot
            self.is_stand = False
            charging_time = self.capacity / self.e_self_charge
            moving_time = distance.euclidean(self.start, self.end) / self.velocity
            self.end_time = time_stem + moving_time + charging_time
        self.check_state()
