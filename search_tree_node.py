'''
SearchTreeNode Class used in A* Search
'''

class SearchTreeNode:
    
    ##################################################################
    # Constructor
    ##################################################################
    
    def __init__(self, state, action, parent, totalCost, heuristicCost):
        """
        SearchTreeNodes contain the following information for A*:

        :state: The state represented by the node, a tuple:
        (x, y) = (col, row)
        
        :action: The action used to transition from the parent to this node.
        - The action's value is None if the initial state
        - The action's value will be a string "U", "D", "L", "R" otherwise
        
        :parent: The parent of this node in the search tree.
        - The parent's value is None if the initial state
        - The parent's value is a reference to the parent node otherwise
        
        :totalCost: The total cost of the path of actions that led to this particular state.
        In the notes, we refer to this value as being evaluated through g(n)
        
        :heuristicCost: The heuristic estimate of cost to be incurred from this node to the
        optimal solution
        """
        self.state = state
        self.action = action
        self.parent = parent
        self.totalCost = totalCost
        self.heuristicCost = heuristicCost
    
    
    ##################################################################
    # Methods
    ##################################################################
    
    def evaluation(self):
        return self.totalCost + self.heuristicCost
        
    def __lt__(self, other):
        return self.evaluation() < other.evaluation()
    