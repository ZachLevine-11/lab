##need this in a separate file to get the queue to work
from scipy.cluster.hierarchy import ClusterNode


##gets the node id of a cluster node and creates an expanded node that also holds the p value
class pvalNode(ClusterNode):
    def __init__(self, node, pval=None):
        ##Pass all the information to an updated node
        super().__init__(id=node.id, left=node.left, right=node.right, dist=node.dist, count=node.count)
        self.pval = pval
        self.alpha = None

    @property
    def is_significant(self):
        return self.pval < self.alpha
