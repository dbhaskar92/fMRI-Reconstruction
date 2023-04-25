 
 
class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v 
    
def Text2List(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(item + '\n')
    f.close()
  
 