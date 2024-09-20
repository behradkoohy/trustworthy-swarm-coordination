from welford import Welford

class WelfordR(Welford):
    def __init__(self, elements=None):
        super().__init__(elements=elements)

    def __repr__(self):
        return "(" + str(self.mean) + " " + str(self.var_p) + ")"