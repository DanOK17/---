import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import random
import pickle

# Файл для сохранения результатов игры
SAVE_FILE = 'tic_tac_toe_results.txt'
Q_TABLE_FILE = 'q_table.pkl'  # Файл для сохранения таблицы Q-значений

# Функция для сохранения результатов в более читаемом формате
def save_results(game_data):
    if os.path.exists(SAVE_FILE):
        mode = 'a'
    else:
        mode = 'w'

    with open(SAVE_FILE, mode) as file:
        game_number = sum(1 for line in open(SAVE_FILE)) // 3 + 1
        file.write(f"Игра №{game_number}:\n")
        moves_str = ', '.join([f"{move[0]} на позиции {move[1] + 1}" for move in game_data['moves']])
        file.write(f"Ходы: {moves_str}\n")
        file.write(f"Исход: {game_data['result']}\n\n")

def load_games():
    games = []
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, 'r') as file:
            content = file.read().strip().split("\n\n")
            for game in content:
                lines = game.split("\n")
                if len(lines) >= 3:
                    moves_line = lines[1].replace("Ходы: ", "").strip()
                    moves = []
                    for move_str in moves_line.split(", "):
                        player, position = move_str.split(" на позиции ")
                        moves.append((player, int(position) - 1))
                    result = lines[2].replace("Исход: ", "").strip()
                    games.append({'moves': moves, 'result': result})
    return games

# Класс для Q-learning
class QLearningTicTacToe:
    def __init__(self, player='O', alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = self.load_q_table()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.player = player

    def get_state(self, board):
        return ''.join(board)

    def choose_action(self, state, available_moves):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        else:
            q_values = [self.q_table.get((state, move), 0) for move in available_moves]
            max_q_value = max(q_values)
            best_moves = [move for move, q_value in zip(available_moves, q_values) if q_value == max_q_value]
            return random.choice(best_moves)

    def update_q_table(self, state, action, reward, next_state, next_available_moves):
        next_max_q_value = max([self.q_table.get((next_state, move), 0) for move in next_available_moves]) if next_available_moves else 0
        current_q_value = self.q_table.get((state, action), 0)
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * next_max_q_value - current_q_value)
        self.q_table[(state, action)] = new_q_value

    def check_win_condition(self, board, player):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for combo in win_combinations:
            if board[combo[0]] == board[combo[1]] == player and board[combo[2]] == '':
                return combo[2]
            if board[combo[0]] == board[combo[2]] == player and board[combo[1]] == '':
                return combo[1]
            if board[combo[1]] == board[combo[2]] == player and board[combo[0]] == '':
                return combo[0]
        return None

    def check_double_win_condition(self, board, player):
        """Проверяет наличие множественных линий для победы на следующем ходе."""
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        win_count = 0
        for combo in win_combinations:
            if (board[combo[0]] == player and board[combo[1]] == player and board[combo[2]] == '') or \
               (board[combo[0]] == player and board[combo[2]] == player and board[combo[1]] == '') or \
               (board[combo[1]] == player and board[combo[2]] == player and board[combo[0]] == ''):
                win_count += 1
        return win_count >= 2

    def train(self, games):
        for game in games:
            board = [''] * 9
            state = self.get_state(board)
            for move in game["moves"]:
                player, action = move
                if player == self.player:
                    available_moves = [i for i, x in enumerate(board) if x == '']
                    chosen_action = action
                    board[action] = player
                    next_state = self.get_state(board)
                    reward = 0

                    # Базовое вознаграждение за победу/проигрыш
                    if game["result"] == f"Победил игрок {player}":
                        reward = 1
                    elif game["result"] == f"Победил игрок {'O' if player == 'X' else 'X'}":
                        reward = -1

                    # Дополнительное вознаграждение за создание нескольких линий для победы
                    if self.check_double_win_condition(board, player):
                        reward += 0.5

                    next_available_moves = [i for i, x in enumerate(board) if x == '']
                    self.update_q_table(state, chosen_action, reward, next_state, next_available_moves)
                    state = next_state

    def play_ai_turn(self, board):
        state = self.get_state(board)
        available_moves = [i for i, x in enumerate(board) if x == '']
        block_move = self.check_win_condition(board, 'X')
        if block_move is not None:
            return block_move

        # Попытка создать несколько путей для победы
        for move in available_moves:
            temp_board = board[:]
            temp_board[move] = self.player
            if self.check_double_win_condition(temp_board, self.player):
                return move

        return self.choose_action(state, available_moves)

    def save_q_table(self):
        with open(Q_TABLE_FILE, 'wb') as file:
            pickle.dump(self.q_table, file)

    def load_q_table(self):
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, 'rb') as file:
                return pickle.load(file)
        return {}

# Класс для игры в крестики-нолики
class TicTacToe:
    def __init__(self):
        self.board = [''] * 9
        self.game_active = True
        self.moves = []
        self.q_agent = QLearningTicTacToe()
        previous_games = load_games()
        self.q_agent.train(previous_games)
        self.q_agent.save_q_table()
        self.start_game()

    def start_game(self):
        clear_output()
        self.board = [''] * 9
        self.game_active = True
        self.moves = []
        self.player_symbol = 'X'
        self.ai_symbol = 'O'

        # Выбор за кого играть
        self.selection_box = widgets.RadioButtons(
            options=['X', 'O'],
            description='Выберите символ:',
            disabled=False
        )
        self.start_button = widgets.Button(description='Начать игру')
        self.start_button.on_click(self.select_player)
        display(self.selection_box)
        display(self.start_button)

    def select_player(self, button):
        self.player_symbol = self.selection_box.value
        self.ai_symbol = 'O' if self.player_symbol == 'X' else 'X'
        self.q_agent.player = self.ai_symbol
        self.initialize_board()

        if self.ai_symbol == 'X':
            self.make_ai_move()

    def initialize_board(self):
        clear_output()
        self.buttons = [widgets.Button(description='', layout=widgets.Layout(width='80px', height='80px')) for _ in range(9)]
        self.restart_button = widgets.Button(description='Перезапустить игру', layout=widgets.Layout(width='300px'))
        self.restart_button.on_click(self.restart_game)
        self.save_button = widgets.Button(description='Сохранить результаты', layout=widgets.Layout(width='300px'))
        self.save_button.on_click(self.save_game)
        self.display_game()

    def display_game(self):
        clear_output()
        grid = widgets.GridBox(self.buttons, layout=widgets.Layout(grid_template_columns="repeat(3, 80px)"))
        for i, button in enumerate(self.buttons):
            button.on_click(lambda b, i=i: self.handle_move(i))
        display(grid)
        display(self.restart_button)
        display(self.save_button)

    # Minimax функция для поиска наилучшего хода
    def minimax(self, board, depth, is_maximizing):
        if self.check_winner_in_minimax(board, self.ai_symbol):
            return 1  # ИИ выиграл
        if self.check_winner_in_minimax(board, self.player_symbol):
            return -1  # Игрок выиграл
        if '' not in board:
            return 0  # Ничья

        if is_maximizing:
            best_score = -float('inf')
            for i in range(9):
                if board[i] == '':
                    board[i] = self.ai_symbol
                    score = self.minimax(board, depth + 1, False)
                    board[i] = ''
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(9):
                if board[i] == '':
                    board[i] = self.player_symbol
                    score = self.minimax(board, depth + 1, True)
                    board[i] = ''
                    best_score = min(score, best_score)
            return best_score

    # Найти наилучший ход с помощью minimax
    def best_move(self):
        best_score = -float('inf')
        move = None
        for i in range(9):
            if self.board[i] == '':
                self.board[i] = self.ai_symbol
                score = self.minimax(self.board, 0, False)
                self.board[i] = ''
                if score > best_score:
                    best_score = score
                    move = i
        return move

    # Проверка победителя для minimax
    def check_winner_in_minimax(self, board, player):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for combo in win_combinations:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] == player:
                return True
        return False

    def handle_move(self, index):
        if self.board[index] == '' and self.game_active:
            self.board[index] = self.player_symbol
            self.buttons[index].description = self.player_symbol
            self.moves.append((self.player_symbol, index))

            if self.check_winner():
                self.game_active = False
                messagebox = widgets.Label(f'Игрок {self.player_symbol} выиграл!')
                display(messagebox)
                self.q_agent.train([{'moves': self.moves, 'result': f'Победил игрок {self.player_symbol}'}])
                self.q_agent.save_q_table()
            elif '' not in self.board:
                self.game_active = False
                messagebox = widgets.Label('Ничья!')
                display(messagebox)
            else:
                self.make_ai_move()

    def make_ai_move(self):
      ai_move = self.best_move()  # Использование функции minimax для выбора наилучшего хода
      self.board[ai_move] = self.ai_symbol
      self.buttons[ai_move].description = self.ai_symbol
      self.moves.append((self.ai_symbol, ai_move))

      if self.check_winner():
          self.game_active = False
          messagebox = widgets.Label(f'Игрок {self.ai_symbol} выиграл!')
          display(messagebox)
          self.q_agent.train([{'moves': self.moves, 'result': f'Победил игрок {self.ai_symbol}'}])
          self.q_agent.save_q_table()
      elif '' not in self.board:
          self.game_active = False
          messagebox = widgets.Label('Ничья!')
          display(messagebox)

    def check_winner(self):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for combo in win_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != '':
                return True
        return False

    def save_game(self, button):
        result = 'Ничья!' if '' not in self.board else f'Победил игрок {self.ai_symbol}' if not self.game_active and self.check_winner() else f'Победил игрок {self.player_symbol}'
        save_results({'moves': self.moves, 'result': result})

    def restart_game(self, button):
        self.start_game()

# Запуск игры
TicTacToe()
