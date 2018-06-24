class GameState(object):
    def __init__(self):
        self.Board = None
        self.Player = None
        self.PreviousPlayer = None
        return 

    def Copy(self):
        raise NotImplementedError

    def LegalActions(self):
        raise NotImplementedError

    def LegalActionShape(self):
        raise NotImplementedError

    def ApplyAction(self, action):
        raise NotImplementedError
    
    def Winner(self, prevAction = None):
        raise NotImplementedError

    def NumericRepresentation(self):
        raise NotImplementedError