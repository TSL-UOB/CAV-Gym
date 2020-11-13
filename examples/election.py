from library.actions import TargetOrientation
from examples.agents import ElectionPedestrianAgent


class Election:
    def __init__(self, env, agents):
        assert len(env.actors) == len(agents)
        self.env = env
        self.agents = agents

        self.electorate = [i for i, agent in enumerate(self.agents) if isinstance(agent, ElectionPedestrianAgent)]
        self.active_player = None

    def focal_joint_action(self, joint_action_vote, focal_player):
        assert focal_player in self.electorate
        for i in self.electorate:
            if i != focal_player:  # only the winner gets to cross
                self.agents[i].reset()  # tell the agent their orientation action will not be executed
                velocity_action_id, _ = joint_action_vote[i]  # allow the agent's velocity action to be executed
                joint_action_vote[i] = velocity_action_id, TargetOrientation.NOOP.value  # reset the agent's orientation action
        return joint_action_vote

    def result(self, previous_state, joint_action_vote):
        assert len(self.agents) == len(previous_state) == len(joint_action_vote)

        if self.active_player and not self.agents[self.active_player].crossing_action:
            target_orientation = previous_state[self.active_player][7]
            active_orientation = target_orientation is not None
            if not active_orientation:
                previous_active_player = self.active_player
                self.active_player = None  # previous winner has finished crossing
                return self.focal_joint_action(joint_action_vote, previous_active_player)  # previous_active_player needs to execute final action

        if self.active_player:
            return self.focal_joint_action(joint_action_vote, self.active_player)

        winner = None

        voters = []
        for i in self.electorate:
            _, orientation_action_id = joint_action_vote[i]
            orientation_action = TargetOrientation(orientation_action_id)
            if orientation_action is not TargetOrientation.NOOP:  # agent has voted to cross
                voters.append(i)

        if voters:
            joint_winners = []
            winning_distance = float("inf")
            for i in voters:
                distance = self.env.actors[i].state.position.distance(self.env.ego.state.position)
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
