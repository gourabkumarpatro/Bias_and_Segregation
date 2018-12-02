import numpy as np
from scipy.stats import truncnorm

from heapq import heapify, heappush, heappop
import random

class Member :
    """
    Member in a community.
    """
    def __init__(self, ID) :
        self.opinion = np.random.normal(0, 1/2)
        self.epsilon = np.random.random() * .5
        self.ID = ID

class Community :
    """
    A community of members.
    """
    def __init__(self, n, gamma=1.0, alpha=0.5, activity=10000) :
        # Initialize parameters.
        assert(n>0)
        assert(0.0 <= alpha <= 1.0)
        assert(activity > 0)
        assert(gamma>=0.0)
        self.n = n
        self.alpha = alpha
        self.activity = activity
        self.gamma = gamma
        self.epsilon = 10**(-10)
        self.threshold = 10**(-10)

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
        total = 0
        for timeline in self.timelines :
            while len(timeline) > 0 and timeline[-1] > cutoff_time :
                timeline.pop()
            timeline.reverse()
            total += len(timeline)

        # Generate schedule.
        self.schedule = [ (self.timelines[idx][-1], idx) for idx in range(n)
                            if len(self.timelines[idx]) > 0 ]
        heapify(self.schedule)

    def interactions(self) :
        """
        Returns the time and member ID for the next interaction.
        """
        while len(self.schedule) > 0 : # All done.
            timestamp, ID = heappop(self.schedule)
            self.timelines[ID].pop()
            if self.timelines[ID] :
                heappush(self.schedule, (self.timelines[ID][-1], ID))
            yield timestamp, ID

    def exchange_opinions(self, x, y) :
        """
        Make x interact with y - two members of this community.
        Their interaction changes their opinion by a
        ratio of alpha towards one another.
        """
        opx, opy = x.opinion, y.opinion
        x.opinion = self.alpha * opy + (1-self.alpha) * opx
        y.opinion = self.alpha * opx + (1-self.alpha) * opy

    def sample_interaction(self, x) :
        """
        Sample the member with whom the member x interacts.
        x interacts with any of the persons having difference of opinion
        less than x.epsilon with a probability given by the power law :
        p(x,y) = d(x,y)^-gamma / sum(z)(d(x,z)^-gamma)
        """
        candidates = [ (abs(member.opinion-x.opinion)**(-self.gamma), member.ID)
            for member in self.members
            if ( (self.epsilon < abs(member.opinion-x.opinion) < x.epsilon) and (member.ID != x.ID) ) ]
        norm = sum(candidate[0] for candidate in candidates)
        random_variate = random.random() * norm
        for candidate in candidates :
            if candidate[0] > random_variate :
                return candidate[1]
            random_variate -= candidate[0]

    def simulate(self) :
        plot_o = [ [member.opinion] for member in self.members ]
        plot_t = [ [0] for _ in range(self.n) ]

        prev_state = np.array([member.opinion for member in self.members])
        curr_state = np.array([member.opinion for member in self.members])
        mavg_diff  = np.zeros_like(curr_state)

        prev_time = 0
        counter, total = 0, sum(len(timeline) for timeline in self.timelines)
        for __time, x in self.interactions() :
            counter+=1
            print('%8d/%8d'%(counter, total), end='\r')
            y = self.sample_interaction(self.members[x])
            if y is None : continue # Isolated opinion
            self.exchange_opinions(self.members[x], self.members[y])
            plot_o[x].append(self.members[x].opinion)
            plot_o[y].append(self.members[y].opinion)
            plot_t[x].append(__time)
            plot_t[y].append(__time)
            consensus_time = __time
            prev_state = curr_state.copy()
            curr_state[x] = self.members[x].opinion
            curr_state[y] = self.members[y].opinion
            diff = (curr_state - prev_state) / (__time - prev_time + self.epsilon)
            mavg_diff = mavg_diff * .99 + diff * .01
            prev_time = __time
            # print( np.sum(mavg_diff**2) )
            if np.sum( mavg_diff**2 ) < self.threshold : break
        return plot_o, plot_t, prev_time
