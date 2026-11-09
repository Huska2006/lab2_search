import heapq
import json
import math
from typing import List, Tuple, Dict, Set, Optional

INF = float("inf")
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右


# =========================================================
# 题目一：迷宫寻路 A*
# =========================================================
def astar_maze(start: Tuple[int, int],
               goal: Tuple[int, int],
               grid: List[List[str]],
               use_heuristic: bool = True):
    """
    返回 (path, expanded)
    path: List[Tuple[int,int]]，无解返回 []
    expanded: 扩展节点数，用于实验对照
    """
    R, C = len(grid), len(grid[0])

    def h(p):
        if not use_heuristic:
            return 0
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    open_heap = [(h(start), 0, start)]
    g_score = {start: 0}
    came_from = {}
    expanded = 0

    while open_heap:
        f, gcur, cur = heapq.heappop(open_heap)
        if gcur != g_score.get(cur, INF):
            continue
        expanded += 1

        if cur == goal:
            break

        r, c = cur
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if grid[nr][nc] == "X":
                continue

            nxt = (nr, nc)
            ng = gcur + 1
            if ng < g_score.get(nxt, INF):
                g_score[nxt] = ng
                came_from[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))

    if goal not in g_score:
        return [], expanded

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = came_from[cur]
    path.append(start)
    path.reverse()
    return path, expanded


# =========================================================
# 题目二：加权地形 A*
# =========================================================
def cell_cost(ch):
    """
    进入某格需要付出的代价。
    X: 不可通行
    E: 终点，按题目视为 0
    S/A/B/T1...: 视为 1
    数字字符/数字: 地形权值
    """
    if ch == "X":
        return None
    if ch == "E":
        return 0
    if isinstance(ch, int):
        return ch
    if ch in ("S", "A", "B"):
        return 1
    if isinstance(ch, str) and ch.startswith("T"):
        return 1
    return int(ch)


def astar_weighted(start: Tuple[int, int],
                   goal: Tuple[int, int],
                   grid: List[List[str]],
                   use_heuristic: bool = True):
    """
    返回 (min_cost, expanded)
    无解 min_cost = -1
    移动代价 = 进入邻居格的 cell_cost
    起点代价不计，终点 E 代价为 0
    """
    R, C = len(grid), len(grid[0])

    def h(p):
        if not use_heuristic:
            return 0
        # 题目二除 E 外最小进入代价为 1，曼哈顿距离可采纳
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    open_heap = [(h(start), 0, start)]
    g_score = {start: 0}
    expanded = 0

    while open_heap:
        f, gcur, cur = heapq.heappop(open_heap)
        if gcur != g_score.get(cur, INF):
            continue
        expanded += 1

        if cur == goal:
            return gcur, expanded

        r, c = cur
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            step = cell_cost(grid[nr][nc])
            if step is None:
                continue

            nxt = (nr, nc)
            ng = gcur + step
            if ng < g_score.get(nxt, INF):
                g_score[nxt] = ng
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))

    return -1, expanded


# =========================================================
# 题目三：Minimax 与 Alpha-Beta
# =========================================================
class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


def build_tree_from_dict(obj) -> TreeNode:
    """
    支持两种 JSON：
    1. 叶子 {"value": 3}
    2. 非叶子 {"children": [...], "is_maximizing_player": true/false}
    注意：递归函数中玩家类型由层数交替决定，根为 MAX。
    """
    if "value" in obj and "children" not in obj:
        return TreeNode(value=obj["value"])
    children = [build_tree_from_dict(ch) for ch in obj.get("children", [])]
    return TreeNode(children=children)


def minimax(node: TreeNode, is_max: bool, counter=None):
    """返回 (score, visited_nodes)"""
    if counter is None:
        counter = [0]
    counter[0] += 1

    if not node.children:
        return node.value, counter[0]

    if is_max:
        best = -INF
        for ch in node.children:
            val, _ = minimax(ch, False, counter)
            best = max(best, val)
        return best, counter[0]
    else:
        best = INF
        for ch in node.children:
            val, _ = minimax(ch, True, counter)
            best = min(best, val)
        return best, counter[0]


def alpha_beta(node: TreeNode, is_max: bool,
               alpha: float = -INF, beta: float = INF,
               counter=None, cuts=None):
    """
    返回 (score, visited_nodes, cut_count)
    子节点从左到右访问。
    """
    if counter is None:
        counter = [0]
    if cuts is None:
        cuts = [0]
    counter[0] += 1

    if not node.children:
        return node.value, counter[0], cuts[0]

    if is_max:
        best = -INF
        for ch in node.children:
            val, _, _ = alpha_beta(ch, False, alpha, beta, counter, cuts)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                cuts[0] += 1
                break
        return best, counter[0], cuts[0]
    else:
        best = INF
        for ch in node.children:
            val, _, _ = alpha_beta(ch, True, alpha, beta, counter, cuts)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                cuts[0] += 1
                break
        return best, counter[0], cuts[0]


def build_sample_tree_q3() -> TreeNode:
    """
    按题目第12页代码示例构建。
    注意示例中 max2 被 min1 和 min2 共用；这里为清晰可复制节点，
    数值结果仍为 5。
    """
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)

    max1 = TreeNode(children=[leaf1, leaf2])
    max2a = TreeNode(children=[leaf3, leaf4])
    max2b = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])

    min1 = TreeNode(children=[max1, max2a])
    min2 = TreeNode(children=[max2b, max3])
    root = TreeNode(children=[min1, min2])
    return root


# =========================================================
# 题目四：竞争环境下的宝藏分配
# =========================================================
TREASURE_MAP = [
    ["A", "1", "1", "X", "2", "1", "T1"],
    ["1", "X", "6", "X", "1", "X", "1"],
    ["1", "1", "T2", "1", "1", "1", "1"], # 把原来的 3 改成 1
    ["X", "1", "X", "8", "X", "1", "X"],  # 把开头改成 X，去掉行号
    ["1", "1", "1", "1", "T3", "1", "1"],
    ["1", "X", "9", "X", "1", "X", "1"],
    ["T4", "1", "1", "X", "1", "1", "B"],
]

TREASURE_VALUE = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
TREASURE_POS = {"T1": (0, 6), "T2": (2, 2), "T3": (4, 4), "T4": (6, 0)}
KEY_POS = {"A": (0, 0), "B": (6, 6), **TREASURE_POS}
KEYS = ["A", "B", "T1", "T2", "T3", "T4"]


def build_distance_matrix():
    """
    6x6 距离矩阵，顺序：A, B, T1, T2, T3, T4
    d[p][q] = 从 p 到 q 的最小移动代价，进入 q 的代价计入，起点不计
    """
    mat = {p: {} for p in KEYS}
    for p in KEYS:
        for q in KEYS:
            if p == q:
                mat[p][q] = 0
            else:
                cost, _ = astar_weighted(KEY_POS[p], KEY_POS[q], TREASURE_MAP, True)
                mat[p][q] = cost
    return mat


def print_distance_matrix(mat):
    print("距离矩阵，顺序：", KEYS)
    print("      " + "".join(f"{k:>6}" for k in KEYS))
    for p in KEYS:
        row = "".join(f"{mat[p][q]:>6}" for q in KEYS)
        print(f"{p:>4}  {row}")


def treasure_gain(pos_key: str, target: str, mat) -> int:
    return TREASURE_VALUE[target] - mat[pos_key][target]


def solve_treasure(H: int, use_alpha_beta: bool = True, verbose: bool = True):
    """
    返回 (best_target, best_value, visited_nodes)
    H：搜索回合深度。H=2 表示 A、B 各走一步后评估；H=4 表示 A、B、A、B。
    """
    mat = build_distance_matrix()
    all_targets = set(TREASURE_VALUE.keys())
    visited = 0

    def ordered(remaining):
        return sorted(remaining, key=lambda x: int(x[1:]))

    def rec(depth, posA, posB, remaining, U, is_A, alpha, beta):
        nonlocal visited
        visited += 1

        if depth >= H or not remaining:
            return U

        if is_A:
            best = -INF
            for t in ordered(remaining):
                new_U = U + treasure_gain(posA, t, mat)
                val = rec(depth + 1, t, posB, remaining - {t}, new_U, False, alpha, beta)
                best = max(best, val)
                if use_alpha_beta:
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break
            return best
        else:
            best = INF
            for t in ordered(remaining):
                new_U = U - treasure_gain(posB, t, mat)
                val = rec(depth + 1, posA, t, remaining - {t}, new_U, True, alpha, beta)
                best = min(best, val)
                if use_alpha_beta:
                    beta = min(beta, best)
                    if beta <= alpha:
                        break
            return best

    # 根节点 A 单独循环，便于记录首步
    best_target = None
    best_value = -INF
    alpha = -INF
    beta = INF

    for t in ordered(all_targets):
        U0 = treasure_gain("A", t, mat)
        val = rec(1, t, "B", all_targets - {t}, U0, False, alpha, beta)
        if val > best_value:
            best_value = val
            best_target = t
        if use_alpha_beta:
            alpha = max(alpha, best_value)

    if verbose:
        print(f"H={H}, use_alpha_beta={use_alpha_beta}, "
              f"best_target={best_target}, best_value={best_value}, visited={visited}")
    return best_target, best_value, visited


# =========================================================
# 样例测试
# =========================================================
def test_q1():
    grid = [
        ["S", " ", " ", "X", " "],
        [" ", "X", " ", " ", " "],
        [" ", "X", " ", " ", " "],
        [" ", "X", " ", " ", " "],
        [" ", "X", " ", " ", " "],
    ]
    start, goal = (0, 0), (3, 4)
    path, expanded = astar_maze(start, goal, grid, True)
    print("题目一路径：", path)
    print("题目一扩展节点数：", expanded)

    path0, expanded0 = astar_maze(start, goal, grid, False)
    print("题目一 h=0 扩展节点数：", expanded0)


def test_q2():
    grid = [
        [5, 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    cost, expanded = astar_weighted((0, 0), (4, 4), grid, True)
    print("题目二最小代价：", cost, "扩展节点数：", expanded)

    cost0, expanded0 = astar_weighted((0, 0), (4, 4), grid, False)
    print("题目二 h=0 最小代价：", cost0, "扩展节点数：", expanded0)


def test_q3():
    root = build_sample_tree_q3()
    score1, n1 = minimax(root, True)
    score2, n2, cuts = alpha_beta(root, True)
    print("题目三 Minimax:", score1, "访问节点:", n1)
    print("题目三 Alpha-Beta:", score2, "访问节点:", n2, "剪枝次数:", cuts)


def test_q4():
    mat = build_distance_matrix()
    print_distance_matrix(mat)
    print("\nMinimax / Alpha-Beta 对照：")
    for H in [2, 4]:
        solve_treasure(H, use_alpha_beta=False)
        solve_treasure(H, use_alpha_beta=True)


if __name__ == "__main__":
    print("=" * 30, "题目一", "=" * 30)
    test_q1()
    print("=" * 30, "题目二", "=" * 30)
    test_q2()
    print("=" * 30, "题目三", "=" * 30)
    test_q3()
    print("=" * 30, "题目四", "=" * 30)
    test_q4()