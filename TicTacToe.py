import numpy as np


class Agent:

    def __init__(self, eps=0.1, alpha=0.5):
        """
        A player of tic tac toe that will learn to play through RL using epslon greedy algorithm
        :param eps: epsilon value for epsilon greedy strategy (probability of random action)
        :param alpha: learning rate
        """
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.state_history = []

    def set_V(self, V):
        """
        initialize the value function
        :param V: a value function to start with
        """
        self.V = V

    def set_symbol(self, symbol):
        """
        give a symbol to play with on the board
        :param symbol:
        """
        self.symbol = symbol

    def set_verbose(self, v):
        """
        print information option e.g. the current board before a move
        :param b: boolean
        """
        self.verbose = v

    def reset_history(self):
        """
        reset the history befor starting a new episode
        :return:
        """
        self.state_history = []

    def take_action(self, env):
        """
        check the board (in env) for valid moves and take action according to the strategy (e.g. epsilon greedy)
        :param env:
        :return:
        """
        r = np.random.rand()
        possible_moves = np.argwhere(env.board == 0)
        if r < self.eps:
            if self.verbose: print("Exploring: Taking random action")
            idx = np.random.choice(range(possible_moves.shape[0]))
            next_move = tuple(possible_moves[idx])
        else:
            pos2val = {}
            next_move = None
            best_value = -1
            for i, j in possible_moves:
                env.board[i, j] = self.symbol
                state = env.get_state()
                env.board[i, j] = 0
                pos2val[(i, j)] = self.V[state]
                if self.V[state] > best_value:
                    best_value = self.V[state]
                    next_move = (i, j)

            if self.verbose:
                print("Exploiting: Taking a greedy action")
                d = {env.x: "x", env.o: "o"}
                for i in range(env.length):
                    print("----------------")
                    for j in range(env.length):
                        if env.is_empty(i, j):
                            print(" %.2f|" % pos2val[(i, j)], end="")
                        else:
                            print(" ", end="")
                            print(d.get(env.board[i][j]) + " |", end="")
                    print("")
                print("----------------")

                # env.draw_board()
                # for (i, j) in sorted(pos2val.keys(), key=lambda x: x[0]):
                #     print(i, j, pos2val[(i, j)])
        # making the move
        env.board[next_move[0]][next_move[1]] = self.symbol

    def update_state_history(self, state):
        """
        add state to the state history - needs the be called after every turn, not only turns of this Agent
        :param state:
        """
        self.state_history.append(state)

    def update(self, env):
        """
        query the env for the latest reward (called at the end of an episode - this is where all the learning happens)
        we want to backtrack our states so that:
        V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
        where V(prev_state) = reward if it's the most current state

        NOTE: we only do this at the end of an episode, this is not the case for all the RL algorithms
        :param env:
        :return:
        """
        reward = env.reward(self.symbol)
        target = reward
        for prev in reversed(self.state_history):
            # prev_value = self.V[prev]
            # value = prev_value + self.alpha*(target - prev_value)
            value = self.V[prev] + self.alpha * (target - self.V[prev])
            self.V[prev] = value
            target = value
        self.reset_history()


class TTTEnv:
    def __init__(self, length=3):
        self.length = length
        self.board = np.zeros((length, length))
        self.o = 1
        self.x = -1
        self.winner = None
        self.ended = False
        self.num_states = 3**(length*length)

    def is_empty(self, i, j):
        return self.board[i, j] == 0.0

    def reward(self, symbol):
        """

        :param symbol: self.x or self.o
        :return:
        """
        if not self.game_over():
            return 0
        return 1 if self.winner == symbol else 0

    def get_state(self):
        """
        returns the state of the game as its int representation [0, S-1] where S is the number of all possible states
        possible symbols {"", "x", "o"}
        S = #symbols ** (length*length)
        note that some states aren't possible but this int representation is simple, we just ignore invalid states
        representations e.g. all board filled with "x"
        :return:
        """
        k = 0  # 0-8
        hash = 0
        switch_dict = {0: 0, self.x: 1, self.o: 2}
        for i in range(self.length):
            for j in range(self.length):
                v = switch_dict[self.board[i, j]]
                hash += (3**k) * v
                k += 1
        return hash

    def draw_board(self):
        """
        visualize the board
        """
        board_string = ""
        symbol_dict = {self.x: "x", self.o: "o"}
        for i in range(self.length):
            for j in range(self.length):
                board_string += "|"
                board_string += symbol_dict.get(self.board[i, j], " ")
            board_string += "|\n"
        print(board_string)

    def _update_and_return(self, symbol):
        self.winner = symbol
        self.ended = True
        return True

    def game_over(self, force_recalc=False):
        """

        :param force_recalc:
        :return:
        """

        if not force_recalc and self.ended:
            return self.ended

        # check rows
        for i in range(self.length):
            for symbol in (self.x, self.o):
                if self.board[i].sum() == symbol*self.length:
                    return self._update_and_return(symbol)
        # check cols
        for j in range(self.length):
            for symbol in (self.x, self.o):
                if self.board[:, j].sum() == symbol * self.length:
                    return self._update_and_return(symbol)

        # check diagonals top left
        for symbol in (self.x, self.o):
            if self.board.trace() == symbol * self.length:
                return self._update_and_return(symbol)

        # check diagonals top right
        for symbol in (self.x, self.o):
            if np.fliplr(self.board).trace() == symbol * self.length:
                return self._update_and_return(symbol)

        # check draw
        if not np.any(self.board == 0):
            self.winner = None
            self.ended = True
            return True

        # game on
        self.winner = None
        return False


class Human:
    def __init__(self):
        pass

    def set_symbol(self, symbol):
        self.symbol = symbol

    def take_action(self, env):
        while True:
            move = input("Enter i,j coordinates for the next move: ")
            i, j = [int(x) for x in move.split(",")]
            if env.is_empty(i, j):
                env.board[i][j] = self.symbol
                break

    def update_state_history(self, state):
        pass

    def update(self, env):
        pass


def get_state_hash_and_winner(env, i=0, j=0):
    resuts = []
    for v in (0, env.x, env.o):
        env.board[i, j] = v
        if j == 2:
            # j goes back to 0, i increases and if i = 2 we are done
            if i == 2:
                # board is full, collect results and return
                state = env.get_state()
                ended = env.game_over(force_recalc=True)
                winner = env.winner
                resuts.append((state, winner, ended))
            else:
                # end of the row next column (i+1), first row (j=0)
                resuts += get_state_hash_and_winner(env, i+1, 0)
        else:
            # increment j, i stays the same
            resuts += get_state_hash_and_winner(env, i, j+1)
        return resuts


def initialize_v(env, state_winner_end_triplates, symbol):
    V = np.zeros(env.num_states)
    for state, winner, is_end in state_winner_end_triplates:
        if is_end:
            v = 1 if winner == symbol else 0
        else:
            v = 0.5
        V[state] = v
    return V


def play_game(p1, p2, env, draw=0):
    # loop until game is over

    current_player = None
    while not env.game_over():
        # alternate between players
        current_player = p1 if current_player != p1 else p2

        # draw current board before next move
        do_draw = draw and (draw == 1 and current_player == p1 or draw == 2 and current_player == p2)
        if do_draw:
            env.draw_board()

        # current player makes its move
        current_player.take_action(env)

        # update state histories
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

    if draw:
        env.draw_board()
        dt = {-1: "x", 1: "o"}
        winner = dt.get(env.winner) if env.winner else "no one"
        print(f"Winner is: {winner}")

    # value function update
    p1.update(env)
    p2.update(env)


if __name__ == "__main__":
    p1 = Agent()
    p2 = Agent()
    env = TTTEnv()
    state_winner_player_triples = get_state_hash_and_winner(env)

    vx = initialize_v(env, state_winner_player_triples, env.x)
    p1.set_V(vx)
    vo = initialize_v(env, state_winner_player_triples, env.o)
    p2.set_V(vo)

    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    T = 1000
    for t in range(T):
        if not (t % 200): print(t)
        play_game(p1, p2, TTTEnv())

    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
        play_game(p1, human, TTTEnv(), draw=2)
        # AI is the first player (p1) to see if it select the center of the board as its first action
        # but it can also be p2

        answer = input("play one more? [Y/n]: ")
        if answer and answer.lower()[0] == "n":
            break
