import numpy as np
from scipy.stats import truncnorm

from heapq import heapify, heappush, heappop

class Member :
    """
    Member in a community.
    """
    def __init__(self, ID) :
        self.opinion = np.random.normal(0.5, 1/6)
        self.epsilon = np.random.random()
        self.ID = ID

class Community :
    """
    A community of members.
    """
    def __init__(self, n, alpha=0.5, activity=1000) :
        # Initialize parameters.
        assert(n>0)
        assert(0.0 <= alpha <= 1.0)
        assert(activity > 0)
        self.n = n
        self.alpha = alpha
        self.activity = activity

        # Instantiate members.
        self.members = [ Member(ID) for ID in range(n) ]

        # Sample timeline.
        mean_times = (5 + truncnorm.rvs(-3, 3, size=self.n)) / 2
        mean_times = 1 / mean_times
        self.timelines = [ list(np.random.exponential(mean_time, self.activity))
                            for mean_time in mean_times ]
        for timeline in self.timelines :
            for i in range(1, activity) :
                timeline[i] += timeline[i-1]

        # Truncate interactions to earliest last interaction time.
        cutoff_time = min(timeline[-1] for timeline in self.timelines)
        for timeline in self.timelines :
            while len(timeline) > 0 and timeline[-1] > cutoff_time :
                timeline.pop()
            timeline.reverse()

        # Generate schedule.
        self.schedule = [ (self.timelines[idx][-1], idx) for idx in range(n) ]
        heapify(self.schedule)

    def interactions(self) :
        """
        Returns the ID of the next interacting member.
        """
        while len(self.schedule) > 0 : # All done.
            _, ID = heappop(self.schedule)
            if len(self.timelines[ID]) > 0 :
                self.timelines[ID].pop()
                heappush(self.schedule, (self.timelines[ID][-1], ID))
            yield ID

    def exchange_opinions(self, x, y) :
        """
        Make x interact with y - two members of this community.
        Their interaction changes their opinion by a
        ratio of alpha towards one another.
        """
        opx, opy = x.opinion, y.opinion
        x.opinion = self.alpha * opy + (1-self.alpha) * opx
        y.opinion = self.alpha * opx + (1-self.alpha) * opy


community = Community(300, alpha=0.3)
