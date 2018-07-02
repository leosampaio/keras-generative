import threading
from abc import ABCMeta, abstractmethod


class Metric(object, metaclass=ABCMeta):
    metrics_by_name = {}

    def __init__(self):
        self.thread = threading.Thread()
        self.thread.start()

    def compute_in_parallel(self, input_data):
        self.thread.join()  # wait for previous computation to finish
        self.thread = threading.Thread(target=self.computation_worker,
                                       args=(input_data,))
        self.thread.start()

    @abstractmethod
    def computation_worker(self, input_data):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def get_data_for_plot(self):
        pass

    @abstractmethod
    def is_ready_for_plot(self):
        pass

    @property
    @abstractmethod
    def plot_type(self):
        pass

    @property
    @abstractmethod
    def input_type(self):
        pass

    @property
    @abstractmethod
    def last_value_repr(self):
        return ''

    @classmethod
    def build_metric_by_name(cls, metric_name):
        return cls.metrics_by_name[metric_name]()


class HistoryMetric(Metric):
    plot_type = 'lines'

    def __init__(self):
        super().__init__()
        self.history = []
        self.last_value = None

    def computation_worker(self, input_data):
        try:
            result = self.compute(input_data)
        except Exception as e:
            print("Exception while computing metrics: {}".format(repr(e)))
            if last_value is None:
                result = 0
            else:
                result = self.last_value
        self.last_value = result
        self.history.append(result)

    def get_data_for_plot(self):
        return self.history

    def is_ready_for_plot(self):
        return bool(self.history)

    @property
    def last_value_repr(self):
        if self.last_value is not None:
            return str(self.last_value)
        else:
            return 'waiting'


class ProjectionMetric(Metric):
    plot_type = 'scatter'

    def __init__(self):
        super().__init__()
        self.current_projection = None

    def computation_worker(self, input_data):
        try:
            result = self.compute(input_data)
        except Exception as e:
            print("Exception while computing metrics: {}".format(repr(e)))
            if not self.current_projection:
                result = [[0.], [0.]]
            else:
                result = self.current_projection
        self.current_projection = result

    def get_data_for_plot(self):
        return self.current_projection

    def is_ready_for_plot(self):
        return self.current_projection is not None

    @property
    def last_value_repr(self):
        if self.current_projection is not None:
            return 'plotted'
        else:
            return 'waiting'


class ImageSamplesMetric(Metric):
    plot_type = 'image-grid'

    def __init__(self):
        super().__init__()
        self.current_images = None

    def computation_worker(self, input_data):
        try:
            result = self.compute(input_data)
        except Exception as e:
            print("Exception while computing metrics: {}".format(repr(e)))
        self.current_images = result

    def get_data_for_plot(self):
        return self.current_images

    def is_ready_for_plot(self):
        return self.current_images is not None

    @property
    def last_value_repr(self):
        if self.current_images is not None:
            return 'image-grid'
        else:
            return 'waiting'
