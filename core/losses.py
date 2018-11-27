import numpy as np
import pickle
from keras import backend as K


class Loss(object):

    def __init__(self, weight=1.0, weight_control_type='none', pivot_control_epoch=1):
        self.last_value = 100.
        self.history = []
        self.weight = weight
        self.weight_control_type = weight_control_type
        self.pivot_control_epoch = pivot_control_epoch
        self.weight_from_last_significant_change = 0.
        self.weight_history = []
        self.current_weight = 0.
        self.backend = K.variable(0)
        self.did_set_initial_weights = False

        allowed_weight_control_types = ['inc', 'dec', 'hold', 'halt', 'none',
                                        'hold-inc', 'hold-dec', 'zero', 'skip']
        if weight_control_type not in allowed_weight_control_types:
            raise ValueError("Weight control type must be one of {}".format(allowed_weight_control_types))

    @classmethod
    def from_control_string(cls, control_string):
        """
        Builds a loss object from control string defined in the format:
        loss_name:weight:control_type:pivot_epoch
        """
        attributes = control_string.split(':')
        if len(attributes) == 2:  # we have only weight and name
            _, weight = attributes
            w = float(weight)
            if w == 0:
                return cls(weight=float(weight), weight_control_type='zero')
            else:
                return cls(weight=float(weight))
        elif len(attributes) == 4:
            _, weight, lcontrol, pivot_epoch = attributes
            return cls(weight=float(weight),
                       weight_control_type=lcontrol,
                       pivot_control_epoch=float(pivot_epoch))
        else:
            raise ValueError("Wrong format for loss control string")

    def update_weight_based_on_time(self, current_epoch):
        if self.weight_control_type == 'zero':
            self.backend = 0.
            self.current_weight = 0.
            return 0., 0.
        weighting_factor = np.min((1, (current_epoch) / self.pivot_control_epoch))
        new_weight = 0.0
        if self.weight_control_type == 'inc':
            new_weight = self.weight * weighting_factor
        elif self.weight_control_type == 'dec':
            new_weight = self.weight * (1 - weighting_factor)
        elif self.weight_control_type == 'hold':
            if current_epoch >= self.pivot_control_epoch:
                new_weight = self.weight
            else:
                new_weight = 0.0
        elif self.weight_control_type == 'hold-inc':
            if current_epoch >= self.pivot_control_epoch:
                weighting_factor = np.min((1, (current_epoch - self.pivot_control_epoch) / self.pivot_control_epoch))
                new_weight = self.weight * weighting_factor
            else:
                new_weight = 0.0
        elif self.weight_control_type == 'hold-dec':
            if current_epoch >= self.pivot_control_epoch:
                weighting_factor = np.min((1, (current_epoch - self.pivot_control_epoch) / self.pivot_control_epoch))
                new_weight = self.weight * (1 - weighting_factor)
            else:
                new_weight = 0.0
        elif self.weight_control_type == 'halt':
            if current_epoch >= self.pivot_control_epoch:
                new_weight = 0.0
            else:
                new_weight = self.weight
        elif self.weight_control_type == 'none':
            new_weight = self.weight
        elif self.weight_control_type == 'skip':
            if self.did_set_initial_weights:
                self.current_weight = K.get_value(self.backend)
                return self.current_weight, 0.
            else:
                self.did_set_initial_weights = True
                new_weight = self.weight

        K.set_value(self.backend, new_weight)
        self.current_weight = new_weight
        return new_weight, np.abs(new_weight - self.weight_from_last_significant_change)

    def reset_weight_from_last_significant_change(self):
        if self.weight_control_type != 'zero':
            self.weight_from_last_significant_change = K.get_value(self.backend)

    def update_history(self, new_loss):
        self.last_value = new_loss
        self.history.append(new_loss)
        self.weight_history.append(self.current_weight)

    def update_weight_scalar(self):
        if self.weight_control_type != 'zero':
            self.current_weight = K.get_value(self.backend)

    def get_mean_of_latest(self, n=1000):
        if len(self.history) > n:
            return np.mean(self.history[-n:])
        elif len(self.history) == 0:
            return 0
        else:
            return np.mean(self.history)

    def get_std_of_latest(self, n=1000):
        if len(self.history) > n:
            return np.std(self.history[-n:])
        elif len(self.history) == 0:
            return 0
        else:
            return np.std(self.history)

    def adjust_base_weight(self, x):
        self.weight = x
        K.set_value(self.backend, x)

    def save(self, filepath):
        loss_dict = {
            "last_value": self.last_value,
            "history": self.history,
            "weight": self.weight,
            "weight_control_type": self.weight_control_type,
            "pivot_control_epoch": self.pivot_control_epoch,
            "weight_from_last_significant_change": self.weight_from_last_significant_change,
            "weight_history": self.weight_history,
            "current_weight": self.current_weight,
            "did_set_initial_weights": self.did_set_initial_weights,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(loss_dict, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            loss_dict = pickle.load(f)
            self.last_value = loss_dict["last_value"]
            self.history = loss_dict["history"]
            self.weight = loss_dict["weight"]
            self.weight_control_type = loss_dict["weight_control_type"]
            self.pivot_control_epoch = loss_dict["pivot_control_epoch"]
            self.weight_from_last_significant_change = loss_dict["weight_from_last_significant_change"]
            self.weight_history = loss_dict["weight_history"]
            self.current_weight = loss_dict["current_weight"]
            self.did_set_initial_weights = loss_dict["did_set_initial_weights"]
            if self.backend:
                K.set_value(self.backend, self.current_weight)
