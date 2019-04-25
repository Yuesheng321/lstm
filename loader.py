class Dataloader:
    def __init__(self, x, y=None, batch_size=16):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        # self.shuffer = shuffer

    def __iter__(self):
        N = self.x.shape[0]
        if self.y is None:
            return iter(self.x[i:i+self.batch_size] for i in range(0, N, self.batch_size))
        else:
            return iter([self.x[i:i+self.batch_size], self.y[i:i+self.batch_size]] for i in range(0, N, self.batch_size))
