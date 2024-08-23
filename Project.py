from time import time
import numpy as np
TIMEOUT = 4.995
from numpy import ndarray
from typing import NamedTuple, Literal, Tuple
from typing import List, Callable
import random
import math
import numpy as np
from tkinter import *
from typing import Optional
from time import sleep
from PIL import ImageColor
from numpy import ndarray
from typing import NamedTuple, Literal, Tuple, List, Callable, Optional



class GameStatus (NamedTuple):
    board_status: ndarray
    row_status: ndarray
    col_status: ndarray
    player1_turn: bool

class DotsAndBoxesGameAction(NamedTuple):
    action_type: Literal["row", "col"]
    position: Tuple[int, int]

class Player:
    def get_dots_and_boxes_action(self, state: GameStatus ) -> DotsAndBoxesGameAction:
        raise NotImplementedError()

class OpponentEvaluator(Player):
    def __init__(self):
        self.is_player1 = True
        self.global_time = 0
    def get_dots_and_boxes_action(self, state: GameStatus ) -> DotsAndBoxesGameAction:
        self.is_player1 = state.player1_turn
        selected_action: DotsAndBoxesGameAction = None
        self.global_time = time() + TIMEOUT
        row_not_filled = np.count_nonzero(state.row_status == 0)
        column_not_filled = np.count_nonzero(state.col_status == 0)
        for i in range(row_not_filled + column_not_filled):
            try:
                actions = self.generate_dots_and_boxes_actions(state)
                utilities = np.array([self.get_minimax_value(
                    state=self.get_dots_and_boxes_result(state, action), max_depth=i + 1) for action in actions])
                index = np.random.choice(
                    np.flatnonzero(utilities == utilities.max()))
                selected_action = actions[index]
            except TimeoutError:
                break
        return selected_action
    def generate_dots_and_boxes_actions(self, state: GameStatus ) -> List[DotsAndBoxesGameAction]:
        row_positions = self.generate_positions(state.row_status)
        col_positions = self.generate_positions(state.col_status)
        actions: List[DotsAndBoxesGameAction] = []
        for position in row_positions:
            actions.append(DotsAndBoxesGameAction("row", position))
        for position in col_positions:
            actions.append(DotsAndBoxesGameAction("col", position))
        return actions
    def generate_positions(self, matrix: np.ndarray):
        [ny, nx] = matrix.shape
        positions: List[tuple[int, int]] = []
        for y in range(ny):
            for x in range(nx):
                if matrix[y, x] == 0:
                    positions.append((x, y))
        return positions
    def get_dots_and_boxes_result(self, state: GameStatus , action: DotsAndBoxesGameAction) -> GameStatus :
        type = action.action_type
        x, y = action.position
        new_state = GameStatus (
            state.board_status.copy(),
            state.row_status.copy(),
            state.col_status.copy(),
            state.player1_turn,
        )
        player_modifier = -1 if new_state.player1_turn else 1
        is_point_scored = False
        val = 1
        [ny, nx] = new_state.board_status.shape
        if y < ny and x < nx:
            new_state.board_status[y, x] = (
                abs(new_state.board_status[y, x]) + val
            ) * player_modifier
            if abs(new_state.board_status[y, x]) == 4:
                is_point_scored = True
        if type == "row":
            new_state.row_status[y, x] = 1
            if y > 0:
                new_state.board_status[y - 1, x] = (
                    abs(new_state.board_status[y - 1, x]) + val
                ) * player_modifier
                if abs(new_state.board_status[y - 1, x]) == 4:
                    is_point_scored = True
        elif type == "col":
            new_state.col_status[y, x] = 1
            if x > 0:
                new_state.board_status[y, x - 1] = (
                    abs(new_state.board_status[y, x - 1]) + val
                ) * player_modifier
                if abs(new_state.board_status[y, x - 1]) == 4:
                    is_point_scored = True
        new_state = new_state._replace(
            player1_turn=not (new_state.player1_turn ^ is_point_scored)
        )
        return new_state
    def get_minimax_value(
            self,
            state: GameStatus ,
            depth: int = 0,
            max_depth: int = 0,
            alpha: float = -np.inf,
            beta: float = np.inf,
            ) -> float:
        if time() >= self.global_time:
            raise TimeoutError()
        if self.terminal_test_dots_and_boxes(state) or depth == max_depth:
            return self.get_dots_and_boxes_utility(state)
        if self.is_player1 == state.player1_turn:
            return self.max_value(state, depth, max_depth, alpha, beta)
        else:
            return self.min_value(state, depth, max_depth, alpha, beta)

    def max_value(
            self,
            state: GameStatus ,
            depth: int,
            max_depth: int,
            alpha: float,
            beta: float,
            ) -> float:
        value = -np.inf
        actions = self.generate_dots_and_boxes_actions(state)
        for action in actions:
            value = max(
            value,
            self.get_minimax_value(
                self.get_dots_and_boxes_result(state, action),
                depth=depth + 1,
                max_depth=max_depth,
                alpha=alpha,
                beta=beta
            ),
            )
            alpha = max(alpha, value)
            if beta <= alpha:
                break
            return value

    def min_value(self,
                  state: GameStatus ,
                  depth: int,
                  max_depth: int,
                  alpha: float,
                  beta: float,
                  ) -> float:
        value = np.inf
        actions = self.generate_dots_and_boxes_actions(state)
        for action in actions:
            value = min(
            value,
            self.get_minimax_value(
                self.get_dots_and_boxes_result(state, action),
                depth=depth + 1,
                max_depth=max_depth,
                alpha=alpha,
                beta=beta
                ),
        )
            beta = min(beta, value)
            if beta <= alpha:
                break
            return value

    def terminal_test_dots_and_boxes(self, state: GameStatus ) -> bool:
        return np.all(state.row_status == 1) and np.all(state.col_status == 1)
    def get_dots_and_boxes_utility(self, state: GameStatus ) -> float:
        [ny, nx] = state.board_status.shape
        utility = 0
        box_won = 0
        box_lost = 0
        for y in range(ny):
            for x in range(nx):
                if self.is_player1:
                    if state.board_status[y, x] == -4:
                        utility += 1
                        box_won += 1
                    elif state.board_status[y, x] == 4:
                        utility -= 1
                        box_lost += 1
                else:
                    if state.board_status[y, x] == -4:
                        utility -= 1
                        box_lost += 1
                    elif state.board_status[y, x] == 4:
                        utility += 1
                        box_won += 1
        if self.chain_count_dots_and_boxes(state) % 2 == 0 and self.is_player1:
            utility += 1
        elif self.chain_count_dots_and_boxes(state) % 2 != 0 and not self.is_player1:
            utility += 1
        if box_won >= 5:
            utility = np.inf
        elif box_lost >= 5:
            utility = -np.inf
        return utility
    def chain_count_dots_and_boxes(self, state: GameStatus ) -> int:
        chain_count_dots_and_boxes = 0
        chain_list: List[List[int]] = []
        for box_num in range(9):
            flag = False
            for chain in chain_list:
                if box_num in chain:
                    flag = True
                    break
            if not flag:
                chain_list.append([box_num])
                self.add_chain_dots_and_boxes(state, chain_list, box_num)
        for chain in chain_list:
            if len(chain) >= 3:
                chain_count_dots_and_boxes += 1
        return chain_count_dots_and_boxes
    def add_chain_dots_and_boxes(self, state: GameStatus , chain_list: List[List[int]], box_num):
        neighbors_num = [box_num - 1, box_num - 3, box_num + 1, box_num + 3]
        for idx in range(len(neighbors_num)):
            if (
                neighbors_num[idx] < 0
                or neighbors_num[idx] > 8
                or (idx % 2 == 0 and neighbors_num[idx] // 3 != box_num // 3)
            ):
                continue
            flag = False
            for chain in chain_list:
                if neighbors_num[idx] in chain:
                    flag = True
                    break
            if not flag and idx % 2 == 0:
                reference = max(box_num, neighbors_num[idx])
                if not state.col_status[reference // 3][reference % 3]:
                    chain_list[-1].append(neighbors_num[idx])
                    self.add_chain_dots_and_boxes(state, chain_list, neighbors_num[idx])
            if not flag and idx % 2 != 0:
                reference = max(box_num, neighbors_num[idx])
                if not state.row_status[reference // 3][reference % 3]:
                    chain_list[-1].append(neighbors_num[idx])
                    self.add_chain_dots_and_boxes(state, chain_list, neighbors_num[idx])



TIMEOUT = 4.995
class LocalSearchPlayer(Player):
    def __init__(
        self,
        end_temperature: float = 0,
        schedule: Callable[[int], float] = lambda t: math.e ** (-t / 100),
        precision: float = 1e-100,
    ) -> None:
        self.end_temperature = end_temperature
        self.schedule = schedule
        self.precision = precision
        self.is_player1 = True
        self.global_time = 0
    def get_dots_and_boxes_action(self, state: GameStatus ) -> DotsAndBoxesGameAction:
        self.is_player1 = state.player1_turn
        current = self.get_random_action(state)
        start_time = 1
        self.global_time = time() + TIMEOUT
        while True:
            current_temperature = self.schedule(start_time)
            if abs(current_temperature - self.end_temperature) <= self.precision or time() >= self.global_time:
                break
            next = self.get_random_action(state)
            delta = self.get_value(state, next) - \
                self.get_value(state, current)
            if delta > 0 or random.random() < math.e ** (delta / current_temperature):
                current = next
            start_time += 1
        return current
    def get_random_action(self, state: GameStatus ) -> DotsAndBoxesGameAction:
        actions = self.generate_dots_and_boxes_actions(state)
        return random.choice(actions)
    def generate_dots_and_boxes_actions(self, state: GameStatus ) -> List[DotsAndBoxesGameAction]:
        row_positions = self.generate_positions(state.row_status)
        col_positions = self.generate_positions(state.col_status)
        actions: List[DotsAndBoxesGameAction] = []
        for position in row_positions:
            actions.append(DotsAndBoxesGameAction("row", position))
        for position in col_positions:
            actions.append(DotsAndBoxesGameAction("col", position))
        return actions
    def generate_positions(self, matrix: np.ndarray):
        [ny, nx] = matrix.shape
        positions: List[tuple[int, int]] = []
        for y in range(ny):
            for x in range(nx):
                if matrix[y, x] == 0:
                    positions.append((x, y))
        return positions
    def get_dots_and_boxes_result(self, state: GameStatus , action: DotsAndBoxesGameAction) -> GameStatus :
        type = action.action_type
        x, y = action.position
        new_state = GameStatus (
            state.board_status.copy(),
            state.row_status.copy(),
            state.col_status.copy(),
            state.player1_turn,
        )
        player_modifier = -1 if new_state.player1_turn else 1
        is_point_scored = False
        val = 1
        [ny, nx] = new_state.board_status.shape
        if y < ny and x < nx:
            new_state.board_status[y, x] = (
                abs(new_state.board_status[y, x]) + val
            ) * player_modifier
            if abs(new_state.board_status[y, x]) == 4:
                is_point_scored = True
        if type == "row":
            new_state.row_status[y, x] = 1
            if y > 0:
                new_state.board_status[y - 1, x] = (
                    abs(new_state.board_status[y - 1, x]) + val
                ) * player_modifier
                if abs(new_state.board_status[y - 1, x]) == 4:
                    is_point_scored = True
        elif type == "col":
            new_state.col_status[y, x] = 1
            if x > 0:
                new_state.board_status[y, x - 1] = (
                    abs(new_state.board_status[y, x - 1]) + val
                ) * player_modifier
                if abs(new_state.board_status[y, x - 1]) == 4:
                    is_point_scored = True
        new_state = new_state._replace(
            player1_turn=not (new_state.player1_turn ^ is_point_scored)
        )
        return new_state
    def get_value(self, state: GameStatus , action: DotsAndBoxesGameAction) -> float:
        new_state = self.get_dots_and_boxes_result(state, action)
        [ny, nx] = new_state.board_status.shape
        utility = 0
        box_won = 0
        box_lost = 0
        for y in range(ny):
            for x in range(nx):
                if self.is_player1:
                    if new_state.board_status[y, x] == -4:
                        utility += 1
                        box_won += 1
                    elif new_state.board_status[y, x] == 4 or abs(new_state.board_status[y, x]) == 3:
                        utility -= 1
                        box_lost += 1
                else:
                    if new_state.board_status[y, x] == -4 or abs(new_state.board_status[y, x]) == 3:
                        utility -= 1
                        box_lost += 1
                    elif new_state.board_status[y, x] == 4:
                        utility += 1
                        box_won += 1
        if self.chain_count_dots_and_boxes(new_state) % 2 == 0 and self.is_player1:
            utility += 1
        elif self.chain_count_dots_and_boxes(new_state) % 2 != 0 and not self.is_player1:
            utility += 1
        if box_won >= 5:
            utility = np.inf
        elif box_lost >= 5:
            utility = -np.inf
        return utility
    def chain_count_dots_and_boxes(self, state: GameStatus ) -> int:
        chain_count_dots_and_boxes = 0
        chain_list: List[List[int]] = []
        for box_num in range(9):
            flag = False
            for chain in chain_list:
                if box_num in chain:
                    flag = True
                    break
            if not flag:
                chain_list.append([box_num])
                self.add_chain_dots_and_boxes(state, chain_list, box_num)
        for chain in chain_list:
            if len(chain) >= 3:
                chain_count_dots_and_boxes += 1
        return chain_count_dots_and_boxes
    def add_chain_dots_and_boxes(self, state: GameStatus , chain_list: List[List[int]], box_num):
        neighbors_num = [box_num - 1, box_num - 3, box_num + 1, box_num + 3]
        for idx in range(len(neighbors_num)):
            if (
                neighbors_num[idx] < 0
                or neighbors_num[idx] > 8
                or (idx % 2 == 0 and neighbors_num[idx] // 3 != box_num // 3)
            ):
                continue
            flag = False
            for chain in chain_list:
                if neighbors_num[idx] in chain:
                    flag = True
                    break
            if not flag and idx % 2 == 0:
                reference = max(box_num, neighbors_num[idx])
                if not state.col_status[reference // 3][reference % 3]:
                    chain_list[-1].append(neighbors_num[idx])
                    self.add_chain_dots_and_boxes(state, chain_list, neighbors_num[idx])
            if not flag and idx % 2 != 0:
                reference = max(box_num, neighbors_num[idx])
                if not state.row_status[reference // 3][reference % 3]:
                    chain_list[-1].append(neighbors_num[idx])
                    self.add_chain_dots_and_boxes(state, chain_list, neighbors_num[idx])
class RandomPlayer(Player):
    def get_dots_and_boxes_action(self, state: GameStatus ) -> DotsAndBoxesGameAction:
        all_row_marked = np.all(state.row_status == 1)
        all_col_marked = np.all(state.col_status == 1)
        if not (all_row_marked or all_col_marked):
            return self.get_random_action(state)
        elif all_row_marked:
            return self.get_random_col_action(state)
        else:
            return self.get_random_row_action(state)
    def get_random_action(self, state: GameStatus ) -> DotsAndBoxesGameAction:
        if random.random() < 0.5:
            return self.get_random_row_action(state)
        else:
            return self.get_random_col_action(state)
    def get_random_row_action(self, state: GameStatus ) -> DotsAndBoxesGameAction:
        position = self.get_random_position_with_zero_value(state.row_status)
        return DotsAndBoxesGameAction("row", position)
    def get_random_position_with_zero_value(self, matrix: np.ndarray):
        [ny, nx] = matrix.shape
        x = -1
        y = -1
        valid = False
        while not valid:
            x = random.randrange(0, nx)
            y = random.randrange(0, ny)
            valid = matrix[y, x] == 0
        return (x, y)
    def get_random_col_action(self, state: GameStatus ) -> DotsAndBoxesGameAction:
        position = self.get_random_position_with_zero_value(state.col_status)
        return DotsAndBoxesGameAction("col", position)


size_of_board = 600
number_of_dots = 4

symbol_size = (size_of_board / 3 - size_of_board / 8) / 2
symbol_thickness = 50
dot_color = "#FFD700"  
player1_color = "#00FF00"  
player1_color_light = "#90EE90" 
player2_color = "#4169E1"  
player2_color_light = "#87CEEB" 
Green_color = "#228B22" 

dot_width = 0.25 * size_of_board / number_of_dots
edge_width = 0.1 * size_of_board / number_of_dots
distance_between_dots = size_of_board / (number_of_dots)
BOT_TURN_INTERVAL_MS = 100
LEFT_CLICK = "<Button-1>"






DOT_COLOR = "#7BC043"
WELCOME_BG_COLOR = "#F9F8F5"
WELCOME_TEXT_COLOR = "#333333"

class WelcomeScreen:
    def __init__(self):
        self.window = Tk()
        self.window.title("Welcome to Dots and Boxes")
        self.canvas = Canvas(self.window, width=size_of_board, height=size_of_board, bg=WELCOME_BG_COLOR)
        self.canvas.pack()
        self.show_welcome_message()
    
    def show_welcome_message(self):
        self.canvas.create_text(
            size_of_board / 2, size_of_board / 3,
            text="Dots and Boxes Game",
            font=("Helvetica", 40, "bold"),
            fill=WELCOME_TEXT_COLOR
        )
        self.canvas.create_text(
            size_of_board / 2, size_of_board / 2,
            text="Click anywhere to continue",
            font=("Helvetica", 20),
            fill=WELCOME_TEXT_COLOR
        )
        self.canvas.bind("<Button-1>", self.close_welcome_screen)
    
    def close_welcome_screen(self, event):
        self.window.destroy()
class Dots_and_Boxes:
    def __init__(self, bot1: Optional[Player] = None, bot2: Optional[Player] = None):
        self.window = Tk()
        self.window.title("Dots and Boxes")
        self.canvas = Canvas(
            self.window, width=size_of_board, height=size_of_board)
        self.canvas.pack()
        self.player1_starts = True
        self.refresh_board()
        self.bot1 = bot1
        self.bot2 = bot2
        self.play_again()
    def play_again(self):
        self.refresh_board()
        self.board_status = np.zeros(
            shape=(number_of_dots - 1, number_of_dots - 1))
        self.row_status = np.zeros(shape=(number_of_dots, number_of_dots - 1))
        self.col_status = np.zeros(shape=(number_of_dots - 1, number_of_dots))
        self.pointsScored = False
        self.player1_starts = not self.player1_starts
        self.player1_turn = not self.player1_starts
        self.reset_board = False
        self.turntext_handle = []
        self.already_marked_boxes = []
        self.display_turn_text()
        self.turn()
    def mainloop(self):
        self.window.mainloop()
    def is_grid_occupied(self, logical_position, type):
        x = logical_position[0]
        y = logical_position[1]
        occupied = True
        if type == "row" and self.row_status[y][x] == 0:
            occupied = False
        if type == "col" and self.col_status[y][x] == 0:
            occupied = False
        return occupied
    def convert_grid_to_logical_position(self, grid_position):
        grid_position = np.array(grid_position)
        position = (grid_position - distance_between_dots / 4) // (
            distance_between_dots / 2
        )
        type = False
        logical_position = []
        if position[1] % 2 == 0 and (position[0] - 1) % 2 == 0:
            x = int((position[0] - 1) // 2)
            y = int(position[1] // 2)
            logical_position = [x, y]
            type = "row"
        elif position[0] % 2 == 0 and (position[1] - 1) % 2 == 0:
            y = int((position[1] - 1) // 2)
            x = int(position[0] // 2)
            logical_position = [x, y]
            type = "col"
        return logical_position, type
    def pointScored(self):
        self.pointsScored = True
    def mark_box(self):
        boxes = np.argwhere(self.board_status == -4)
        for box in boxes:
            if list(box) not in self.already_marked_boxes and list(box) != []:
                self.already_marked_boxes.append(list(box))
                color = player1_color_light
                self.shade_box(box, color)
        boxes = np.argwhere(self.board_status == 4)
        for box in boxes:
            if list(box) not in self.already_marked_boxes and list(box) != []:
                self.already_marked_boxes.append(list(box))
                color = player2_color_light
                self.shade_box(box, color)
    def update_board(self, type, logical_position):
        x = logical_position[0]
        y = logical_position[1]
        val = 1
        playerModifier = 1
        if self.player1_turn:
            playerModifier = -1
        if y < (number_of_dots - 1) and x < (number_of_dots - 1):
            self.board_status[y][x] = (
                abs(self.board_status[y][x]) + val
            ) * playerModifier
            if abs(self.board_status[y][x]) == 4:
                self.pointScored()
        if type == "row":
            self.row_status[y][x] = 1
            if y >= 1:
                self.board_status[y - 1][x] = (
                    abs(self.board_status[y - 1][x]) + val
                ) * playerModifier
                if abs(self.board_status[y - 1][x]) == 4:
                    self.pointScored()
        elif type == "col":
            self.col_status[y][x] = 1
            if x >= 1:
                self.board_status[y][x - 1] = (
                    abs(self.board_status[y][x - 1]) + val
                ) * playerModifier
                if abs(self.board_status[y][x - 1]) == 4:
                    self.pointScored()
    def is_gameover(self):
        return (self.row_status == 1).all() and (self.col_status == 1).all()
    from time import sleep
    def make_edge(self, type, logical_position):
        if type == "row":
            start_x = distance_between_dots / 2 + logical_position[0] * distance_between_dots
            end_x = start_x + distance_between_dots
            start_y = distance_between_dots / 2 + logical_position[1] * distance_between_dots
            end_y = start_y
        elif type == "col":
            start_y = distance_between_dots / 2 + logical_position[1] * distance_between_dots
            end_y = start_y + distance_between_dots
            start_x = distance_between_dots / 2 + logical_position[0] * distance_between_dots
            end_x = start_x
        if self.player1_turn:
            color = player1_color
        else:
            color = player2_color
        for i in range(11):
            progress = i / 10.0
            x = start_x + (end_x - start_x) * progress
            y = start_y + (end_y - start_y) * progress
            self.canvas.create_line(
                start_x, start_y, x, y, fill=color, width=edge_width, tags="temp_line"
                )
            self.canvas.update()  
            sleep(0.05) 
            self.canvas.delete("temp_line")  
        self.canvas.create_line(
        start_x, start_y, end_x, end_y, fill=color, width=edge_width
        )
    def display_gameover(self):
        player1_score = len(np.argwhere(self.board_status == -4))
        player2_score = len(np.argwhere(self.board_status == 4))
        if player1_score > player2_score:
            text = "Winner: Player 1 "
            color = player1_color
        elif player2_score > player1_score:
            text = "Winner: AI "
            color = player2_color
        else:
            text = "Its a tie"
            color = "gray"
        self.canvas.delete("all")
        self.canvas.create_text(
            size_of_board / 2,
            size_of_board / 3,
            font="cmr 60 bold",
            fill=color,
            text=text,
        )
        score_text = "Scores \n"
        self.canvas.create_text(
            size_of_board / 2,
            5 * size_of_board / 8,
            font="cmr 40 bold",
            fill=Green_color,
            text=score_text,
        )
        score_text = "Player 1 : " + str(player1_score) + "\n"
        score_text += "AI: " + str(player2_score) + "\n"
        self.canvas.create_text(
            size_of_board / 2,
            3 * size_of_board / 4,
            font="cmr 30 bold",
            fill=Green_color,
            text=score_text,
        )     
        self.reset_board = True
        score_text = "Click to play again \n"
        self.canvas.create_text(
            size_of_board / 2,
            15 * size_of_board / 16,
            font="cmr 20 bold",
            fill="gray",
            text=score_text,
        )
    def refresh_board(self):
        for i in range(number_of_dots):
            x = i * distance_between_dots + distance_between_dots / 2
            self.canvas.create_line(
                x,
                distance_between_dots / 2,
                x,
                size_of_board - distance_between_dots / 2,
                fill="red",
                dash=(2, 2),
            )
            self.canvas.create_line(
                distance_between_dots / 2,
                x,
                size_of_board - distance_between_dots / 2,
                x,
                fill="red",
                dash=(2, 2),
            )
        for i in range(number_of_dots):
            for j in range(number_of_dots):
                start_x = i * distance_between_dots + distance_between_dots / 2
                end_x = j * distance_between_dots + distance_between_dots / 2
                self.canvas.create_oval(
                    start_x - dot_width / 2,
                    end_x - dot_width / 2,
                    start_x + dot_width / 2,
                    end_x + dot_width / 2,
                    fill=dot_color,
                    outline=dot_color,
                )
    def display_turn_text(self):
        text = "Current turn: "
        if self.player1_turn:
            text += "Player1"
            color = player1_color
        else:
            text += "AI"
            color = player2_color

        self.canvas.delete(self.turntext_handle)
        self.turntext_handle = self.canvas.create_text(
            size_of_board - 5 * len(text),
            size_of_board - distance_between_dots / 8,
            font="cmr 15 bold",
            text=text,
            fill=color,
        )
    def shade_box(self, box, color):
        start_x = (
            distance_between_dots / 2 + box[1] *
            distance_between_dots + edge_width / 2
        )
        start_y = (
            distance_between_dots / 2 + box[0] *
            distance_between_dots + edge_width / 2
        )
        end_x = start_x + distance_between_dots - edge_width
        end_y = start_y + distance_between_dots - edge_width
        self.canvas.create_rectangle(
            start_x, start_y, end_x, end_y, fill=color, outline=""
        )
    def display_turn_text(self):
        text = "Current turn: "
        if self.player1_turn:
            text += "Player1"
            color = player1_color
        else:
            text += "AI"
            color = player2_color

        self.canvas.delete(self.turntext_handle)
        self.turntext_handle = self.canvas.create_text(
            size_of_board - 5 * len(text),
            size_of_board - distance_between_dots / 8,
            font="cmr 15 bold",
            text=text,
            fill=color,
        )
    def click(self, event):
        if not self.reset_board:
            grid_position = [event.x, event.y]
            logical_position, valid_input = self.convert_grid_to_logical_position(
                grid_position
            )
            self.update(valid_input, logical_position)
        else:
            self.canvas.delete("all")
            self.play_again()
            self.reset_board = False
    def update(self, valid_input, logical_position):
        if valid_input and not self.is_grid_occupied(logical_position, valid_input):
            self.window.unbind(LEFT_CLICK)
            self.update_board(valid_input, logical_position)
            self.make_edge(valid_input, logical_position)
            self.mark_box()
            self.refresh_board()
            self.player1_turn = (
                not self.player1_turn if not self.pointsScored else self.player1_turn
            )
            self.pointsScored = False

            if self.is_gameover():
                self.display_gameover()
                self.window.bind(LEFT_CLICK, self.click)
            else:
                self.display_turn_text()
                self.turn()
    def turn(self):
        current_bot = self.bot1 if self.player1_turn else self.bot2
        if current_bot is None:
            self.window.bind(LEFT_CLICK, self.click)
        else:
            self.window.after(BOT_TURN_INTERVAL_MS, self.bot_turn, current_bot)
    def bot_turn(self, bot: Player):
        action = bot.get_dots_and_boxes_action(
            GameStatus (
                self.board_status.copy(),
                self.row_status.copy(),
                self.col_status.copy(),
                self.player1_turn,
            )
        )
        self.update(action.action_type, action.position)
if __name__ == "__main__":
    welcome_screen = WelcomeScreen()
    welcome_screen.window.mainloop()
    game = Dots_and_Boxes(
        None,
        LocalSearchPlayer()
    )
    game.mainloop()


