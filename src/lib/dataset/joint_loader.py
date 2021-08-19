import random

class JointIterator:
    def __init__(self, iter1, iter2, dataset1, dataset2):
        self.iter1 = iter1
        self.iter2 = iter2
        self.num_steps = [5, 5]
        self.loader_ind = 0
        self.counter = self.num_steps[self.loader_ind]
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __next__(self):
        if self.counter == 0:
            ind = random.randint(0, 1)
            self.loader_ind = ind
            self.counter = self.num_steps[ind]

        if self.loader_ind == 0:
            result = next(self.iter1, None)
            if result is None:
                self.iter1 = iter(self.dataset1)
                result = next(self.iter1, None)
                if result is None:
                    raise StopIteration
        else:
            result = next(self.iter2, None)
            if result is None:
                self.iter2 = iter(self.dataset2)
                result = next(self.iter2, None)
                if result is None:
                    raise StopIteration

        self.counter -= 1

        return result

class JointLoader:

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset = dataset1.dataset

    def __iter__(self):
        return JointIterator(iter(self.dataset1), iter(self.dataset2), self.dataset1, self.dataset2)

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)
