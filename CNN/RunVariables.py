class Epoch():
    def __init__(self, id, loss, acc, start_time):
        self.id = id
        self.loss = loss
        self.acc = acc
        self.start_time = start_time

class Run():
    def __init__(self, params, id, data, loss, acc, start_time):
        self.params = params
        self.id = id
        self.data = data
        self.loss = loss
        self.acc = acc
        self.start_time = start_time