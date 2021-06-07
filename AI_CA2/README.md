<h1 style = "font-size: 30px; text-align: center;">AI Games Hands On</h1>
<h2 style = "font-size: 25px; text-align: center; color: #666">Name: Sepehr Ghobadi</h2>
<h2 style = "font-size: 25px; text-align: center; color: #666">Student Id: 810098009</h2>
<h4 style="text-align: center">Spring 1400</h4>

# Goal Of Project

In this project, we create an agent that plays blackjack with another pre-written agent in a deterministic environment. the algorithm used for finding agent's best decisions is MiniMax. first we play with basic minimax algorithm and then we use a method called Alpha-Beta Pruning to speed up search and prune search tree.



### __there are some changes in project's initial template:__
- __a method named minimax was added to Blacksin class for building minimax search tree.__
starting from current state as a Max node we try to maximize players expected reward. we check all possible values and expand current node with depth-first search by simulating all moves on copies of game environment, then we pass game to next level of tree where is a Min level. in Min nodes we simulate opponents moves by trying to minimize game's result as -1 is win for opponent and 1 is win for player. this tree continues to grow to leafs where two players has stopped. each leaf is a final state of game where winner is decided and then we backtrack to parent nodes.

   with Alpha-Beta pruning we can prune unnecessary branches of search tree. for example in Min node where opponent is trying to minimize game result if in some branches the opponent's agent reaches to a result having -1 value it is unnecessary to check other branchs as this result is optimum. the same applies for Max nodes and branches that result in player win. of course this is a basic explanation of Alpha-Beta pruning. in minimax function alpha and beta values get passed from higher level nodes and pruning is applied as it is designed in real algorithm.


- __a 'log' parameter was added to some functions in both classes to make log prints optional.__

- __a boolean field called 'Prune' was added to Blacksin class making Alpha-Beta pruning optional in order to compare performance of search with and without pruning.__


```python
import random
import time
from copy import deepcopy
```


```python
class Player:
    def __init__(self, name, num_of_cards):
        """
        The base player class of the game
        Inputs
        -----------
        name = (str) player's name
        num_of_cards = (int) number of cards in the deck
        """
        self.name = name
        self.deck_count = num_of_cards
        self.target = self.deck_count * 2 - 1
        self.cards = []
        self.erases_remaining = self.deck_count // 5
        self.has_stopped = False

    def draw_card(self, card):
        """
        draws a card, and adds it to player cards
        Input
        -------------
        card: (int) the card to be added
        """
        self.cards.append(card)

    def print_info(self):
        """
        prints info of the player
        """
        print(f"{self.name}'s cards: ", end='')
        for c in self.cards:
            print(f'{c}, ', end='')
        print(f'sum: {sum(self.cards)}')
    
    def get_margin(self):
        """
        returns the margin left to target by the player
        Output
        ----------
        (int) margin to target
        """
        return self.target - sum(self.cards)
    
    def cpu_play(self, seen_cards, deck, enemies_cards):
        """
        The function for cpu to play the game
        Inputs
        ----------
        seen_cards:     (list of ints) the cards that have been seen until now
        deck:           (list of ints) the remaining playing deck of the game
        enemies_cards:  (list of ints) the cards that the enemy currently has.
        Output
        ----------
        (str) a command given to the game
        
        """
        if (len(deck) > 0):
            next_card_in_deck = deck[0]
        else:
            next_card_in_deck = 0
        if (len(deck) > 1):
            next_enemy_card_in_deck = deck[1]
        else:
            next_enemy_card_in_deck = 0
        amount_to_target = self.target - sum(self.cards)
        amount_with_next_card = self.target - (sum(self.cards) + next_card_in_deck)
        enemies_amount_to_target = self.target - sum(enemies_cards)
        enemies_amount_with_next_card = self.target - (sum(enemies_cards) + next_enemy_card_in_deck)
        _stop_condition = amount_to_target < next_card_in_deck and self.erases_remaining <= 0
        _draw_condition_1 = next_card_in_deck != 0
        _draw_condition_2 = amount_with_next_card >= 0
        _erase_condition = self.erases_remaining > 0
        _erase_self_condition = amount_to_target < 0
        _erase_opponent_condition_or = enemies_amount_to_target < (self.target // 7)
        _erase_opponent_condition_or_2 = enemies_amount_with_next_card < (self.target // 7) 
        _erase_opponent_condition_or_3 = enemies_amount_with_next_card <= amount_with_next_card
        _erase_opponent_condition_or_4 = enemies_amount_to_target <= amount_to_target
        _erase_opponent_condition = _erase_opponent_condition_or or _erase_opponent_condition_or_2 or _erase_opponent_condition_or_3
        _erase_opponent_condition = _erase_opponent_condition or _erase_opponent_condition_or_4 
        if (_stop_condition):
            return 'stop'
        elif (_draw_condition_1 and _draw_condition_2):
            return 'draw'
        elif(_erase_self_condition and _erase_condition):
            return 'erase_self'
        elif(_erase_opponent_condition and _erase_condition):
            return 'erase_opponent'
        else:
            return 'stop'
    
    def erase(self, target, log=True):
        """
        erases the last card of the target
        Input
        ---------
        target: (Player obj) the player whos last card is about to be erased
        """
        if (len(target.cards) == 0):
            if log:
                print(f'{target.name} has no more eraseble cards!')
            return
        if (self.erases_remaining > 0):
            self.erases_remaining -= 1
            card = target.cards.pop(-1)
            if log:
                print(f'{self.name} erased {card} from {target.name}\'s deck!')
            return
        if log:
            print(f'{self.name} has no more erases remaining!')

    def get_player_cards(self):
        return self.cards

    def get_erases_remained(self):
        return self.erases_remaining
```


```python
class Blacksin:
    def __init__(self, prune, deck_count=21):
        """
        The main game class
        Inputs
        -----------
        deck_count = (int) number of cards in the deck
        """
        self.deck_count = deck_count
        self.target = self.deck_count * 2 - 1
        self.player = Player('player', deck_count)
        self.opponent = Player('opponent', deck_count)
        self.deck = self.shuffle_cards()
        self.seen_cards = []
        self.prune = prune
        
    def state_hash(self):
        return ( str(self.player.has_stopped) + "__" + str(self.opponent.has_stopped)+ "__"
                + str(self.player.erases_remaining) + "__" + str(self.opponent.erases_remaining) + "__"
                + str(self.deck) + "__" + str(self.seen_cards) )
    
    def shuffle_cards(self):
        """ 
        shuffles cards for deck creation
        """
        return list(random.sample(range(1, self.deck_count + 1), self.deck_count))

    def draw_card(self):
        """ 
        draws a card from deck, if non is remaining, ends the game.
        """
        if (len(self.deck) > 0):
            card = self.deck.pop(0)
            self.seen_cards.append(card)
            return card
        print('The deck is empty! ending game...')
        self.opponent.has_stopped = True
        self.player.has_stopped = True
        return -1

    def handout_cards(self):
        """ 
        hands out cards to players
        """
        self.player.draw_card(self.draw_card())
        self.opponent.draw_card(self.draw_card())
        self.player.draw_card(self.draw_card())
        self.opponent.draw_card(self.draw_card())
    
    def handle_input(self, _input, player, log=True):
        """ 
        handles input
        Input
        ------------
        _input: (str) input given by the player
        player: (Player obj)the player that is giving the input
        
        """
        if (player is self.player):
            opponent = self.opponent
        else:
            opponent = self.player
        if (_input == 'stop' or _input == 's'):
            player.has_stopped = True
            if log:
                print(f'{player.name} has stopped')
        elif (_input == 'draw' or _input == 'd'):
            card = self.draw_card()
            if (card == -1): return True
            player.draw_card(card)
            if log:
                print(f'{player.name} drawed a card: {card}')
        elif ((_input == 'erase_self' or _input == 'es')):
            player.erase(player, log)
        elif ((_input == 'erase_opponent' or _input == 'eo')):
            player.erase(opponent, log)
        else:
            if log:
                print(_input)
                print('ERROR: unknown command')
            return False
        return True

    def get_player_input(self, log=True):
        test = deepcopy(self)
        your_input = (test.minimax(turn="player", visited={}, alpha=float("-INF"), beta=float('INF'), prune=self.prune))["move"]
        self.handle_input(your_input, self.player, log)
            
    def minimax(self, turn, visited, alpha, beta, prune=False):
        if (self.state_hash()+"__"+turn) in visited:
            return visited[self.state_hash()+"__"+turn]
        if (self.player.has_stopped and self.opponent.has_stopped):
            return {"value":self.check_for_winners(log=False), "move":None}
        
        if turn == "player":
            if self.player.has_stopped:
                return {"value":self.minimax("opponent", visited, alpha, beta, prune)["value"], "move":None}
            best_move = {"value":float('-INF'), "move":None}
            
            if self.player.erases_remaining > 0:
                if len(self.player.cards) > 0:
                    next_move_copy = deepcopy(self)
                    next_move_copy.handle_input("erase_self", next_move_copy.player, log=False)
                    result = next_move_copy.minimax("opponent", visited, alpha, beta, prune)
                    if result["value"] > best_move["value"]:
                        best_move["value"] = result["value"]
                        best_move["move"] = "erase_self"
                    if best_move["value"] >= beta and prune==True:
                        return best_move
                    alpha = max(alpha, result["value"])
                if len(self.opponent.cards) > 0:
                    next_move_copy = deepcopy(self)
                    next_move_copy.handle_input("erase_opponent", next_move_copy.player, log=False)
                    result = next_move_copy.minimax("opponent", visited, alpha, beta, prune)
                    if result["value"] > best_move["value"]:
                        best_move["value"] = result["value"]
                        best_move["move"] = "erase_opponent"
                    if best_move["value"] >= beta and prune==True:
                        return best_move
                    alpha = max(alpha, result["value"])
                        
            next_move_copy = deepcopy(self)
            next_move_copy.handle_input("stop", next_move_copy.player, log=False)
            result = next_move_copy.minimax("opponent", visited, alpha, beta, prune)
            if result["value"] > best_move["value"]:
                best_move["value"] = result["value"]
                best_move["move"] = "stop"
            if best_move["value"] >= beta and prune==True:
                return best_move
            alpha = max(alpha, result["value"])
 
            if len(self.deck) > 0:
                next_move_copy = deepcopy(self)
                next_move_copy.handle_input("draw", next_move_copy.player, log=False)
                result = next_move_copy.minimax("opponent", visited, alpha, beta, prune)
                if result["value"] > best_move["value"]:
                    best_move["value"] = result["value"]
                    best_move["move"] = "draw"
            if best_move["value"] >= beta and prune==True:
                return best_move
            alpha = max(alpha, result["value"])

            visited[self.state_hash()+"__player"] = best_move
            
            return best_move
                    
        if turn == "opponent":
            if self.opponent.has_stopped:
                return {"value":self.minimax("player", visited, alpha, beta, prune)["value"], "move":None}
            best_move = {"value":float('INF'), "move":None}
            
            if self.opponent.erases_remaining > 0:
                if len(self.opponent.cards) > 0:
                    next_move_copy = deepcopy(self)
                    next_move_copy.handle_input("erase_self", next_move_copy.opponent, log=False)
                    result = next_move_copy.minimax("player", visited, alpha, beta, prune)
                    if result["value"] < best_move["value"]:
                        best_move["value"] = result["value"]
                    if best_move["value"] <= alpha and prune==True:
                        return best_move
                    beta = min(beta, result["value"])
                if len(self.player.cards) > 0:
                    next_move_copy = deepcopy(self)
                    next_move_copy.handle_input("erase_opponent", next_move_copy.opponent, log=False)
                    result = next_move_copy.minimax("player", visited, alpha, beta, prune)
                    if result["value"] < best_move["value"]:
                        best_move["value"] = result["value"]
                    if best_move["value"] <= alpha and prune==True:
                        return best_move
                    beta = min(beta, result["value"])

            next_move_copy = deepcopy(self)
            next_move_copy.handle_input("stop", next_move_copy.opponent, log=False)
            result = next_move_copy.minimax("player", visited, alpha, beta, prune)
            if result["value"] < best_move["value"]:
                best_move["value"] = result["value"]
            if best_move["value"] <= alpha and prune==True:
                return best_move
            beta = min(beta, result["value"])
 
            if len(self.deck) > 0:
                next_move_copy = deepcopy(self)
                next_move_copy.handle_input("draw", next_move_copy.opponent, log=False)
                result = next_move_copy.minimax("player", visited, alpha, beta, prune)
                if result["value"] < best_move["value"]:
                    best_move["value"] = result["value"]
            if best_move["value"] <= alpha and prune==True:
                return best_move
            beta = min(beta, result["value"])
                
            visited[self.state_hash()+"__opponent"] = best_move
            return best_move  
    
    
    def opponent_play(self, log=True):
        """
        function for opponent to play it's turn
        """
        try:
            opponent_input = self.opponent.cpu_play(self.seen_cards, self.deck, self.player.cards)
        except:
            opponent_input = 'stop'
        self.handle_input(opponent_input, self.opponent, log)

    def check_for_winners(self, log=True):
        """
        checks for winners.
        Output
        -----------
        (int) returns 1 if player wins, 0 if draw and -1 if opponent wins
        """
        if log:
            self.opponent.print_info()
            self.player.print_info()
        player_margin = self.player.get_margin()
        opponent_margin = self.opponent.get_margin()
        player_win_condition_1 = opponent_margin < 0 and player_margin >= 0
        player_win_condition_2 = opponent_margin >=0 and player_margin >= 0 and player_margin < opponent_margin
        draw_condition_1 = opponent_margin < 0 and player_margin < 0
        draw_condition_2 = opponent_margin >= 0 and player_margin >= 0 and player_margin == opponent_margin
        opponent_win_condition_1 = player_margin < 0 and opponent_margin >= 0
        opponent_win_condition_2 = opponent_margin >=0 and player_margin >= 0 and player_margin > opponent_margin
        if (player_win_condition_1 or player_win_condition_2):
            if log:
                print(f'the winner is the {self.player.name}!')
            return 1
        elif(draw_condition_1 or draw_condition_2):
            if log:
                print('the game ends in a draw!')
            return 0
        elif(opponent_win_condition_1 or opponent_win_condition_2):
            if log:
                print(f'the winner is the {self.opponent.name}!')
            return -1
        else:
            if log:
                print('an error has accurred! exiting...')
            exit()

    def print_deck(self):
        """
        prints the current deck of the game
        """
        print('full deck: [top] ', end='')
        for i in self.deck:
            print(i, end=' ')
        print('[bottom]')

    def run(self, log=True):
        """
        main function to run the game with
        """
        if log:
            print('\nstarting game... shuffling... handing out cards...')
            print(f'remember, you are aiming for nearest to: {self.target}')
            self.print_deck()
        self.handout_cards()
        turn = 0
        while(not self.player.has_stopped or not self.opponent.has_stopped):
            if (turn == 0):
                if (not self.player.has_stopped):
                    if log:
                        self.opponent.print_info()
                        self.player.print_info()
                    self.get_player_input(log)
            else:
                if (not self.opponent.has_stopped):
                    if log:
                        print('opponent playing...')
                    self.opponent_play(log)
            if log:
                print()
            turn = 1 - turn
        if log:
            print('\nand the winner is...')
        return self.check_for_winners(log)
```

## result of 5 rows


```python
for i in range(5):
    print(f"------------------------------- Game {i+1} started! -------------------------------")
    game = Blacksin(prune=True, deck_count=21)
    result = game.run(log=True)
    print()
    print()
```

    ------------------------------- Game 1 started! -------------------------------
    
    starting game... shuffling... handing out cards...
    remember, you are aiming for nearest to: 41
    full deck: [top] 18 5 9 2 19 11 13 21 14 17 15 16 10 4 7 6 8 1 3 20 12 [bottom]
    opponent's cards: 5, 2, sum: 7
    player's cards: 18, 9, sum: 27
    player erased 9 from player's deck!
    
    opponent playing...
    opponent drawed a card: 19
    
    opponent's cards: 5, 2, 19, sum: 26
    player's cards: 18, sum: 18
    player erased 18 from player's deck!
    
    opponent playing...
    opponent drawed a card: 11
    
    opponent's cards: 5, 2, 19, 11, sum: 37
    player's cards: sum: 0
    player erased 11 from opponent's deck!
    
    opponent playing...
    opponent drawed a card: 13
    
    opponent's cards: 5, 2, 19, 13, sum: 39
    player's cards: sum: 0
    player erased 13 from opponent's deck!
    
    opponent playing...
    opponent has stopped
    
    opponent's cards: 5, 2, 19, sum: 26
    player's cards: sum: 0
    player drawed a card: 21
    
    
    opponent's cards: 5, 2, 19, sum: 26
    player's cards: 21, sum: 21
    player drawed a card: 14
    
    
    opponent's cards: 5, 2, 19, sum: 26
    player's cards: 21, 14, sum: 35
    player has stopped
    
    
    and the winner is...
    opponent's cards: 5, 2, 19, sum: 26
    player's cards: 21, 14, sum: 35
    the winner is the player!
    
    
    ------------------------------- Game 2 started! -------------------------------
    
    starting game... shuffling... handing out cards...
    remember, you are aiming for nearest to: 41
    full deck: [top] 6 15 12 10 17 13 16 7 1 21 4 9 8 3 11 20 14 19 5 18 2 [bottom]
    opponent's cards: 15, 10, sum: 25
    player's cards: 6, 12, sum: 18
    player erased 12 from player's deck!
    
    opponent playing...
    opponent has stopped
    
    opponent's cards: 15, 10, sum: 25
    player's cards: 6, sum: 6
    player erased 6 from player's deck!
    
    
    opponent's cards: 15, 10, sum: 25
    player's cards: sum: 0
    player erased 10 from opponent's deck!
    
    
    opponent's cards: 15, sum: 15
    player's cards: sum: 0
    player erased 15 from opponent's deck!
    
    
    opponent's cards: sum: 0
    player's cards: sum: 0
    player drawed a card: 17
    
    
    opponent's cards: sum: 0
    player's cards: 17, sum: 17
    player has stopped
    
    
    and the winner is...
    opponent's cards: sum: 0
    player's cards: 17, sum: 17
    the winner is the player!
    
    
    ------------------------------- Game 3 started! -------------------------------
    
    starting game... shuffling... handing out cards...
    remember, you are aiming for nearest to: 41
    full deck: [top] 12 14 18 13 20 17 7 10 9 11 21 1 8 6 19 3 4 5 2 16 15 [bottom]
    opponent's cards: 14, 13, sum: 27
    player's cards: 12, 18, sum: 30
    player erased 18 from player's deck!
    
    opponent playing...
    opponent has stopped
    
    opponent's cards: 14, 13, sum: 27
    player's cards: 12, sum: 12
    player erased 12 from player's deck!
    
    
    opponent's cards: 14, 13, sum: 27
    player's cards: sum: 0
    player erased 13 from opponent's deck!
    
    
    opponent's cards: 14, sum: 14
    player's cards: sum: 0
    player erased 14 from opponent's deck!
    
    
    opponent's cards: sum: 0
    player's cards: sum: 0
    player drawed a card: 20
    
    
    opponent's cards: sum: 0
    player's cards: 20, sum: 20
    player has stopped
    
    
    and the winner is...
    opponent's cards: sum: 0
    player's cards: 20, sum: 20
    the winner is the player!
    
    
    ------------------------------- Game 4 started! -------------------------------
    
    starting game... shuffling... handing out cards...
    remember, you are aiming for nearest to: 41
    full deck: [top] 10 19 13 7 16 17 2 12 4 21 14 5 15 8 18 11 20 9 3 6 1 [bottom]
    opponent's cards: 19, 7, sum: 26
    player's cards: 10, 13, sum: 23
    player erased 13 from player's deck!
    
    opponent playing...
    opponent has stopped
    
    opponent's cards: 19, 7, sum: 26
    player's cards: 10, sum: 10
    player erased 10 from player's deck!
    
    
    opponent's cards: 19, 7, sum: 26
    player's cards: sum: 0
    player erased 7 from opponent's deck!
    
    
    opponent's cards: 19, sum: 19
    player's cards: sum: 0
    player erased 19 from opponent's deck!
    
    
    opponent's cards: sum: 0
    player's cards: sum: 0
    player drawed a card: 16
    
    
    opponent's cards: sum: 0
    player's cards: 16, sum: 16
    player has stopped
    
    
    and the winner is...
    opponent's cards: sum: 0
    player's cards: 16, sum: 16
    the winner is the player!
    
    
    ------------------------------- Game 5 started! -------------------------------
    
    starting game... shuffling... handing out cards...
    remember, you are aiming for nearest to: 41
    full deck: [top] 20 18 13 10 1 4 6 11 21 2 15 7 9 5 14 16 3 8 12 19 17 [bottom]
    opponent's cards: 18, 10, sum: 28
    player's cards: 20, 13, sum: 33
    player erased 13 from player's deck!
    
    opponent playing...
    opponent drawed a card: 1
    
    opponent's cards: 18, 10, 1, sum: 29
    player's cards: 20, sum: 20
    player erased 20 from player's deck!
    
    opponent playing...
    opponent drawed a card: 4
    
    opponent's cards: 18, 10, 1, 4, sum: 33
    player's cards: sum: 0
    player drawed a card: 6
    
    opponent playing...
    opponent has stopped
    
    opponent's cards: 18, 10, 1, 4, sum: 33
    player's cards: 6, sum: 6
    player erased 6 from player's deck!
    
    
    opponent's cards: 18, 10, 1, 4, sum: 33
    player's cards: sum: 0
    player erased 4 from opponent's deck!
    
    
    opponent's cards: 18, 10, 1, sum: 29
    player's cards: sum: 0
    player drawed a card: 11
    
    
    opponent's cards: 18, 10, 1, sum: 29
    player's cards: 11, sum: 11
    player drawed a card: 21
    
    
    opponent's cards: 18, 10, 1, sum: 29
    player's cards: 11, 21, sum: 32
    player has stopped
    
    
    and the winner is...
    opponent's cards: 18, 10, 1, sum: 29
    player's cards: 11, 21, sum: 32
    the winner is the player!
    
    


## Final Results and Performance of Algortihm With and Without Alpha-Beta Pruning

### Without Alpha-Beta Pruning


```python
start = time.time()
wins=0
draws=0
for i in range(500):
    game = Blacksin(prune=False, deck_count=21)
    result = game.run(log=False)
    if result==1:
        wins+=1
    if result==0:
        draws+=1
print("In 500 Games :")
print("Wins:", wins)
print("Draws:", draws)
end = time.time()
print(f"time (without pruning): {end-start} seconds")
```

    In 500 Games :
    Wins: 411
    Draws: 9
    time (without pruning): 262.4421548843384 seconds


### With Alpha-Beta Pruning


```python
start = time.time()
wins=0
draws=0
for i in range(500):
    game = Blacksin(prune=True, deck_count=21)
    result = game.run(log=False)
    if result==1:
        wins+=1
    if result==0:
        draws+=1
print("In 500 Games :")
print("Wins:", wins)
print("Draws:", draws)
end = time.time()
print(f"time (without pruning): {end-start} seconds")
```

    In 500 Games :
    Wins: 423
    Draws: 5
    time (without pruning): 97.81658482551575 seconds


# MiniMax in Deterministic and Non-Deterministic Environments

in this game when players can see the unused cards and other players deck and they know the value of cards that ther draw from deck on every decision. so the game is played in an detrministic environment and thus agents can use MiniMax easily by building search tree and visiting all possible states from current state and doing the best action they can do. but in non-deterministic environments where the value of a pulled card from decks is unknown the basic MiniMax cant be used. there are other algorithms for such situatons but we cant also improve MiniMax to perform in non-deterministic games.


### ExpectiMiniMax:

The expectiminimax algorithm is a variation of the minimax algorithm, for use in artificial intelligence systems that play two-player zero-sum games, in which the outcome depends on a combination of the player's skill and __chance__ elements such as dice rolls. In addition to "min" and "max" nodes of the traditional minimax tree, this variant has "chance" ("move by nature") nodes, which take the expected value of a random event occurring. In game theory terms, an expectiminimax tree is the game tree of an extensive-form game of perfect, but incomplete information.

In the traditional minimax method, the levels of the tree alternate from max to min until the depth limit of the tree has been reached. In an expectiminimax tree, the "chance" nodes are interleaved with the max and min nodes. Instead of taking the max or min of the utility values of their children, chance nodes take a weighted average, with the weight being the probability that child is reached.

in this project if the agents cant see the deck cards clearly they can keep track of pulled out cards and also use a probabilty distribution for examining probabilty of drawing each remaining card so they can use this probabilites in expectminimax algorithm and perform good in this version of game.

## Order Of Expanding Nodes

MiniMax is somehow a blind search algorithm as it expands all possible nodes in each state and tries every possible move for reaching to optimum result. so order of taking actions in each states doesnt affect final result in minimax algorithm.

but when we use Alpha-Beta pruning we want to prune branch as maximum as possible in order to reach better performance in case of time and space complexity. in such situations order of expanding nodes can help Alpha-Beta pruning. if we traverse promising branches first the pruning algorithm prune other branches more effectively.

for example in our project where leaf nodes' result is only -1,0 or 1 if in higher branchs we traverse a branch with final '1' result for our agent first, the pruning heuristic after seeing a 'win' branch can prune other 3 branches and it leads to pruning almost 50% of nodes of search tree.

in this game with some observations on different strategies i find out that in each state starting with erasing cards from agents deck or opponent's deck is more promising than drawing card from game's deck and also drawing card is more promising than stopping. the intuition behind this strategy is that stopping too early can give your opponent chance to remove card from your dec and at the same time opponent can draw card so it results in increase of opponents score and decreasing your score. also drawing cards can be dangerous as number of your remaining erases decreases and we get closer to end of game, drawing cards has risk of surpasing the limited 41 score.

