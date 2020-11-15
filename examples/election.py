from examples.agents import ElectionPedestrianAgent


class Election:
    def __init__(self, env, agents):
        assert len(env.bodies) == len(agents)
        self.env = env
        self.agents = agents

        self.electorate = [i for i, agent in enumerate(self.agents) if isinstance(agent, ElectionPedestrianAgent)]
        self.active_player = None

        self.previous_joint_action = None

    def focal_joint_action(self, joint_action_vote, focal_player):
        assert focal_player in self.electorate
        for i in self.electorate:
            if i != focal_player:  # only the winner gets to cross
                self.agents[i].reset()  # tell the agent their orientation action will not be executed
                velocity_action, _ = joint_action_vote[i]  # allow the agent's velocity action to be executed
                joint_action_vote[i] = [velocity_action, 0.0]  # reset the agent's orientation action
        return joint_action_vote

    def result(self, previous_state, joint_action_vote):
        assert len(self.agents) == len(previous_state) == len(joint_action_vote)

        if self.active_player and not self.agents[self.active_player].crossing:
            self.active_player = None

        if self.active_player:
            return self.focal_joint_action(joint_action_vote, self.active_player)

        winner = None

        voters = []
        for i in self.electorate:
            if self.agents[i].voting:  # agent has voted to cross
                voters.append(i)

        if voters:
            joint_winners = []
            winning_distance = float("inf")
            for i in voters:
                distance = self.env.bodies[i].state.position.distance(self.env.ego.state.position)
                if distance < winning_distance:
                    joint_winners = [i]
                    winning_distance = distance
                elif distance == winning_distance:
                    joint_winners.append(i)
            winner = self.env.np_random.choice(joint_winners) if joint_winners else None

        if winner:
            self.active_player = winner  # no elections show take place until the winner finishes crossing
            return self.focal_joint_action(joint_action_vote, self.active_player)
        else:
            return joint_action_vote
