import numpy as np

EMPTY, BLACK, WHITE = 0, 1, -1
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

BOARD_SIZE = 8

def generate_score_table(size):
    table = np.full((size, size), 3)
    table[0, :] = 5; table[-1, :] = 5; table[:, 0] = 5; table[:, -1] = 5
    if size > 4:
        table[1, :] = -5; table[-2, :] = -5; table[:, 1] = -5; table[:, -2] = -5
    if size > 4:
        for i in [2, -3]:
            for j in [0, 1, 2, -3, -2, -1]:
                if 0 <= i < size and 0 <= j < size:
                    table[i, j] = 15 if j in [2, -3] else -5
                    table[j, i] = 15 if j in [2, -3] else -5
    if size > 2:
        table[1, 1] = -80; table[1, -2] = -80; table[-2, 1] = -80; table[-2, -2] = -80
        table[0, 1] = -40; table[1, 0] = -40; table[0, -2] = -40; table[1, -1] = -40
        table[-1, 1] = -40; table[-2, 0] = -40; table[-1, -2] = -40; table[-2, -1] = -40
    table[0, 0] = 120; table[0, -1] = 120; table[-1, 0] = 120; table[-1, -1] = 120
    return table

SCORE_TABLE = generate_score_table(BOARD_SIZE)
TRANSPOSITION_TABLE = {}

def board_to_hash(board):
    return board.tobytes()

def init_board():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    mid = BOARD_SIZE // 2
    board[mid - 1, mid - 1] = WHITE
    board[mid - 1, mid] = BLACK
    board[mid, mid - 1] = BLACK
    board[mid, mid] = WHITE
    global TRANSPOSITION_TABLE
    TRANSPOSITION_TABLE.clear()
    return board

def in_bounds(x, y):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

def get_flips(board, x, y, player):
    if board[x, y] != EMPTY:
        return []
    flips = []
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        line = []
        while 0 <= nx < 8 and 0 <= ny < 8 and board[nx, ny] == -player:
            line.append((nx, ny))
            nx += dx; ny += dy
        if 0 <= nx < 8 and 0 <= ny < 8 and board[nx, ny] == player and line:
            flips.extend(line)
    return flips

def get_legal_moves(board, player):
    return [(x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE) if get_flips(board, x, y, player)]

def make_move(board, x, y, player):
    flips = get_flips(board, x, y, player)
    if not flips:
        return False
    board[x, y] = player
    for fx, fy in flips:
        board[fx, fy] = player
    return True

def count_pieces(board):
    return np.sum(board == BLACK), np.sum(board == WHITE)

def evaluate_board(board, my_color, my_moves_count, opp_moves_count):
    dynamic_table = SCORE_TABLE.copy()
    corners = [
        ((0, 0), [(0, 1), (1, 0), (1, 1)]),
        ((0, -1), [(0, -2), (1, -1), (1, -2)]),
        ((-1, 0), [(-1, 1), (-2, 0), (-2, 1)]),
        ((-1, -1), [(-1, -2), (-2, -1), (-2, -2)])
    ]
    for (cx, cy), neighbors in corners:
        owner = board[cx, cy]
        if owner != EMPTY:
            for nx, ny in neighbors:
                dynamic_table[nx, ny] = 30 if owner == my_color else -100

    score = np.sum(board * dynamic_table) * my_color
    score += (my_moves_count - opp_moves_count) * 15

    stable_score = 0
    for cx, cy in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
        if board[cx, cy] != EMPTY:
            stable_score += 50 if board[cx, cy] == my_color else -50
    score += stable_score
    return score

def alphabeta(board, depth, alpha, beta, player, my_color, best_move_at_parent=None):
    board_hash = (board_to_hash(board), depth, player)
    if board_hash in TRANSPOSITION_TABLE:
        cached_alpha, cached_beta, cached_score, cached_move = TRANSPOSITION_TABLE[board_hash]
        if cached_alpha >= beta: return cached_score, cached_move
        if cached_beta <= alpha: return cached_score, cached_move

    legal_moves = get_legal_moves(board, player)
    opponent_moves = get_legal_moves(board, -player)

    if depth == 0 or (not legal_moves and not opponent_moves):
        my_moves_count = len(legal_moves) if player == my_color else len(opponent_moves)
        opp_moves_count = len(opponent_moves) if player == my_color else len(legal_moves)
        return evaluate_board(board, my_color, my_moves_count, opp_moves_count), None

    if not legal_moves:
        score, _ = alphabeta(board, depth - 1, alpha, beta, -player, my_color)
        return score, None

    if best_move_at_parent and best_move_at_parent in legal_moves:
        legal_moves.remove(best_move_at_parent)
        legal_moves.insert(0, best_move_at_parent)
    else:
        legal_moves.sort(key=lambda m: SCORE_TABLE[m[0], m[1]], reverse=True)

    best_move = None
    original_alpha, original_beta = alpha, beta

    if player == my_color:
        best_score = -float('inf')
        for move in legal_moves:
            temp_board = board.copy()
            make_move(temp_board, move[0], move[1], player)
            score, _ = alphabeta(temp_board, depth - 1, alpha, beta, -player, my_color)
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
            if beta <= alpha: break
        TRANSPOSITION_TABLE[board_hash] = (alpha, original_beta, best_score, best_move)
        return best_score, best_move
    else:
        best_score = float('inf')
        for move in legal_moves:
            temp_board = board.copy()
            make_move(temp_board, move[0], move[1], player)
            score, _ = alphabeta(temp_board, depth - 1, alpha, beta, -player, my_color)
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)
            if beta <= alpha: break
        TRANSPOSITION_TABLE[board_hash] = (original_alpha, beta, best_score, best_move)
        return best_score, best_move

def endgame_alphabeta(board, alpha, beta, player, my_color):
    legal_moves = get_legal_moves(board, player)
    opponent_moves = get_legal_moves(board, -player)

    if not legal_moves and not opponent_moves:
        my_count, opp_count = count_pieces(board) if my_color == BLACK else count_pieces(board)[::-1]
        return my_count - opp_count, None

    if not legal_moves:
        score, _ = endgame_alphabeta(board, alpha, beta, -player, my_color)
        return score, None

    legal_moves.sort(key=lambda m: SCORE_TABLE[m[0], m[1]], reverse=True)

    best_move = None
    if player == my_color:
        best_score = -float('inf')
        for move in legal_moves:
            temp_board = board.copy()
            make_move(temp_board, move[0], move[1], player)
            score, _ = endgame_alphabeta(temp_board, alpha, beta, -player, my_color)
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
            if beta <= alpha: break
        return best_score, best_move
    else:
        best_score = float('inf')
        for move in legal_moves:
            temp_board = board.copy()
            make_move(temp_board, move[0], move[1], player)
            score, _ = endgame_alphabeta(temp_board, alpha, beta, -player, my_color)
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)
            if beta <= alpha: break
        return best_score, best_move

def select_ai_move(board, player):
    empty_count = np.sum(board == EMPTY)
    if empty_count <= 10:
        _, best_move = endgame_alphabeta(board, -float('inf'), float('inf'), player, my_color=player)
        return best_move

    target_depth = 5 if empty_count <= 20 else 4
    best_move = None

    for d in range(1, target_depth + 1):
        _, current_best = alphabeta(board, d, -float('inf'), float('inf'), player, my_color=player, best_move_at_parent=best_move)
        if current_best:
            best_move = current_best

    return best_move
