#帮我生成一个游戏，游戏的背景是80*80的一个画板，然后游戏的A角色可以在画板上画一些形状（可以支持任意形状），然后A角色可以点击隐藏该形状。影藏之后，B用户可以选择查看某个坐标的点，如果该点在A用户画的形状中则显示绿色，如果不再则显示灰色。最后可以显示A角色画的形状然后看看和B用户探索后的形状是否匹配
import pygame
import numpy as np
import random

# 添加网格尺寸常量（在文件开头）
GRID_WIDTH = 80   # 根据实际网格宽度调整
GRID_HEIGHT = 60  # 根据实际网格高度调整

# 初始化pygame
pygame.init()
pygame.key.set_repeat(500, 30)  # 启用键盘重复


# 定义网格大小和单元格大小
GRID_SIZE = 40  # 网格数量保持不变
CELL_SIZE = 25  # 将格子大小从20改为40
MENU_HEIGHT = 80  # 菜单高度保持不变
MENU_FONT_SIZE = 28  # 菜单字体大小
COORD_FONT_SIZE = 16  # 坐标字体大小
BUTTON_Y = 10  # 将按钮的垂直位置从20改为10
COORD_MARGIN = 30  # 坐标轴边距保持不变

# 添加字体初始化
coord_font = pygame.font.SysFont('Arial', COORD_FONT_SIZE)
menu_font = pygame.font.SysFont('Arial', MENU_FONT_SIZE)

# 计算窗口大小
WINDOW_WIDTH = COORD_MARGIN + (GRID_SIZE * CELL_SIZE)  # 左边距 + 网格总宽度
WINDOW_HEIGHT = MENU_HEIGHT + (GRID_SIZE * CELL_SIZE)  # 顶部菜单 + 网格总高度
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("形状探索游戏")

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0) 
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
BLUE = (100, 100, 255)
GRID_COLOR = (200, 200, 200)

# 重新定义实际的绘图区域，排除坐标轴区域
DRAWING_AREA = pygame.Rect(
    COORD_MARGIN,
    MENU_HEIGHT,
    GRID_SIZE * CELL_SIZE,  # 确保完整显示40个格子
    GRID_SIZE * CELL_SIZE
)

# 初始化画板数组
board = np.zeros((80, 80))
discovered = np.zeros((80, 80))

# 游戏状态
drawing = False
start_pos = None
lines = []  # 存储所有画的线
current_role = 'A'  # 当前角色
shapes_hidden = False  # 形状是否隐藏
points = []  # 存储多边形的顶点
drawing_polygon = False  # 是否正在画多边形
polygons = []  # 存储所有画好的多边形
explored_points = []  # 存储B角色探索过的点 [(x, y, is_inside), ...]

# 修改按钮文字和状态变量
current_mode = 'DRAW'  # 当前模式：'DRAW'(画图), 'EXPLORE'(探索), 'VERIFY'(验证)

# 重新计算按钮位置，使间距相等
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 40
TOTAL_BUTTONS = 7

# 计算总间距和按钮占用的总宽度
total_button_width = BUTTON_WIDTH * TOTAL_BUTTONS
remaining_space = WINDOW_WIDTH - total_button_width
button_spacing = remaining_space / (TOTAL_BUTTONS + 1)  # 按钮之间的间距

# 创建字体
font = pygame.font.Font(None, MENU_FONT_SIZE)  # 菜单使用的主字体
coord_font = pygame.font.Font(None, COORD_FONT_SIZE)  # 坐标使用的小字体

# 重新定义按钮位置，y坐标改为BUTTON_Y
button_draw = pygame.Rect(
    button_spacing, 
    BUTTON_Y, 
    BUTTON_WIDTH, 
    BUTTON_HEIGHT
)

button_explore = pygame.Rect(
    button_spacing * 2 + BUTTON_WIDTH, 
    BUTTON_Y, 
    BUTTON_WIDTH, 
    BUTTON_HEIGHT
)

button_verify = pygame.Rect(
    button_spacing * 3 + BUTTON_WIDTH * 2, 
    BUTTON_Y, 
    BUTTON_WIDTH, 
    BUTTON_HEIGHT
)

button_reset = pygame.Rect(
    button_spacing * 4 + BUTTON_WIDTH * 3,
    BUTTON_Y,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)

# 创建坐标输入框
input_box_row = pygame.Rect(
    button_spacing * 5 + BUTTON_WIDTH * 4,
    BUTTON_Y,
    BUTTON_WIDTH // 2 - 5,
    BUTTON_HEIGHT
)

input_box_col = pygame.Rect(
    button_spacing * 5 + BUTTON_WIDTH * 4 + BUTTON_WIDTH // 2 + 5,
    BUTTON_Y,
    BUTTON_WIDTH // 2 - 5,
    BUTTON_HEIGHT
)

# 输入框状态
input_active_row = False  # 行输入框是否激活
input_active_col = False  # 列输入框是否激活
input_text_row = ''
input_text_col = ''
max_input_length = 3  # 限制输入长度为3位数字

# 创建Check按钮
button_check = pygame.Rect(
    button_spacing * 6 + BUTTON_WIDTH * 5,
    BUTTON_Y,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)


# 在全局变量部分添加
text = ""  # 用于存储键盘输入

# 在全局变量部分添加
drawing_mode = 'curve'  # 默认为曲线模式
curve_points = []  # 存储曲线的点
curves = []  # 存储所有已完成的曲线
EXPLORE_CLICK = 'CLICK'
EXPLORE_CURVE = 'CURVE'
EXPLORE_POLYGON = 'POLYGON'
explore_sub_mode = EXPLORE_CLICK  # 默认为点击模式
drawing_explore_curve = False
explore_curve_points = []
explore_curves = []
explore_polygons = []

# 在全局变量部分添加
checked_cells = []  # 存储已检查的方格 [(row, col, is_inside), ...]

# 在全局变量部分添加
input_selected = False  # 输入框内容是否被选中

# 在全局变量部分添加
cursor_pos_row = 0  # 行输入框的光标位置
cursor_pos_col = 0  # 列输入框的光标位置
last_cursor_blink = 0  # 记录光标上次闪烁的时间

# 在全局变量部分添加
explore_curve_colors = []  # 存储每条探索曲线的颜色

def get_random_color():
    """生成随机颜色"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_grid():
    # 填充坐标轴背景
    pygame.draw.rect(screen, (240, 240, 240), (0, MENU_HEIGHT - COORD_MARGIN, WINDOW_WIDTH, COORD_MARGIN))  # x轴区域
    pygame.draw.rect(screen, (240, 240, 240), (0, MENU_HEIGHT, COORD_MARGIN, WINDOW_HEIGHT - MENU_HEIGHT))  # y轴区域
    
    # 绘制网格线
    for x in range(COORD_MARGIN, WINDOW_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, 
                        (x, MENU_HEIGHT), 
                        (x, WINDOW_HEIGHT))
        # 绘制横坐标（x轴）
        if (x - COORD_MARGIN) // CELL_SIZE < GRID_SIZE:
            text = coord_font.render(str((x - COORD_MARGIN) // CELL_SIZE), True, BLACK)
            text_rect = text.get_rect(center=(x + CELL_SIZE//2, MENU_HEIGHT - 15))
            screen.blit(text, text_rect)

    for y in range(MENU_HEIGHT, WINDOW_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, 
                        (COORD_MARGIN, y), 
                        (WINDOW_WIDTH, y))
        # 绘制纵坐标（y轴）
        if (y - MENU_HEIGHT) // CELL_SIZE < GRID_SIZE:
            text = coord_font.render(str((y - MENU_HEIGHT) // CELL_SIZE), True, BLACK)
            text_rect = text.get_rect(midright=(COORD_MARGIN - 5, y + CELL_SIZE//2))
            screen.blit(text, text_rect)

def get_grid_pos(pos):
    """将鼠标坐标转换为网格坐标，包括边缘点"""
    x = (pos[0] - COORD_MARGIN) // CELL_SIZE
    y = (pos[1] - MENU_HEIGHT) // CELL_SIZE
    return max(0, min(GRID_SIZE-1, x)), max(0, min(GRID_SIZE-1, y))

def draw_shape():
    # 绘制A角色画的形状
    for line in lines:
        pygame.draw.line(screen, BLACK, line[0], line[1], 2)

def explore_point(pos):
    # 确保点在绘图区域内
    if not is_in_drawing_area(pos):
        return
    
    # 获取网格坐标
    x, y = get_grid_pos(pos)
    
    # 检查点是否在图形内
    is_inside = False
    
    # 检查多边形
    for polygon in polygons:
        if point_in_polygon(pos, polygon):
            is_inside = True
            break
    
    # 如果不在多边形内，检查曲线
    if not is_inside:
        for curve in curves:
            if len(curve) > 1:
                for i in range(len(curve)-1):
                    if line_intersects_cell(curve[i], curve[i+1], x, y):
                        is_inside = True
                        break
                if is_inside:
                    break
    
    # 绘制探索结果
    rect_x = x * CELL_SIZE + COORD_MARGIN
    rect_y = y * CELL_SIZE + MENU_HEIGHT
    color = GREEN if is_inside else GRAY
    pygame.draw.rect(screen, color, (rect_x, rect_y, CELL_SIZE, CELL_SIZE))
    
    # 添加到探索点列表
    explored_points.append((x, y, is_inside))
    print(f"Explored point at ({x}, {y}): {'Inside' if is_inside else 'Outside'}")

def point_in_polygon(point, vertices):
    """射线法判断点是否在多边形内"""
    x, y = point
    inside = False
    j = len(vertices) - 1
    
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
        
    return inside

def reset_game():
    global points, polygons, curves, curve_points
    global explored_points, explore_curves, explore_polygons, explore_curve_points
    global drawing_curve, drawing_polygon, drawing_explore_curve
    global current_mode, drawing_mode, explore_sub_mode
    global checked_cells, input_text_row, input_text_col

    # 重置所有绘制相关的列表
    points = []              # 当前多边形的点
    polygons = []           # 已完成的多边形
    curves = []             # 已完成的曲线
    curve_points = []       # 当前曲线的点
    
    # 重置所有探索相关的列表
    explored_points = []     # 探索点
    explore_curves = []      # 探索曲线
    explore_polygons = []    # 探索多边形
    explore_curve_points = [] # 当前探索曲线的点
    
    # 重置所有绘制状态
    drawing_curve = False
    drawing_polygon = False
    drawing_explore_curve = False
    
    # 重置模式
    current_mode = 'DRAW'
    drawing_mode = 'curve'
    explore_sub_mode = 'CLICK'
    
    # 重置输入模式和已检查的方格
    checked_cells = []       # 清空已检查的方格
    input_text_row = ''      # 清空行输入框
    input_text_col = ''      # 清空列输入框
    
    print("游戏已重置")  # 调试信息

def draw_menu():
    pygame.draw.rect(screen, BLUE, (0, 0, WINDOW_WIDTH, MENU_HEIGHT))
    
    # 绘制画图按钮
    pygame.draw.rect(screen, GREEN if current_mode == 'DRAW' else GRAY, button_draw)
    text_draw = font.render('DRAW', True, BLACK)
    text_rect = text_draw.get_rect(center=button_draw.center)
    screen.blit(text_draw, text_rect)
    
    # 绘制探索按钮
    pygame.draw.rect(screen, GREEN if current_mode == 'EXPLORE' else GRAY, button_explore)
    text_explore = font.render('EXPLORE', True, BLACK)
    text_rect = text_explore.get_rect(center=button_explore.center)
    screen.blit(text_explore, text_rect)
    
    # 绘制验证按钮
    pygame.draw.rect(screen, GREEN if current_mode == 'VERIFY' else GRAY, button_verify)
    text_verify = font.render('VERIFY', True, BLACK)
    text_rect = text_verify.get_rect(center=button_verify.center)
    screen.blit(text_verify, text_rect)
    
    # 绘制重置按钮
    pygame.draw.rect(screen, (255, 100, 100), button_reset)  # 使用红色表示重置按钮
    text_reset = font.render('RESET', True, BLACK)
    text_rect = text_reset.get_rect(center=button_reset.center)
    screen.blit(text_reset, text_rect)
    # 绘制输入框背景和边框
    pygame.draw.rect(screen, WHITE if input_active_row else GRAY, input_box_row, 2)  # 激活时白色，否则灰色
    pygame.draw.rect(screen, WHITE if input_active_col else GRAY, input_box_col, 2)  # 激活时白色，否则灰色
    
    # 绘制输入框内容（左对齐）
    row_text_surface = font.render(input_text_row, True, BLACK)
    row_text_rect = row_text_surface.get_rect(midleft=(input_box_row.x + 5, input_box_row.centery))  # 左对齐
    screen.blit(row_text_surface, row_text_rect)

    col_text_surface = font.render(input_text_col, True, BLACK)
    col_text_rect = col_text_surface.get_rect(midleft=(input_box_col.x + 5, input_box_col.centery))  # 左对齐
    screen.blit(col_text_surface, col_text_rect)

        
    # 在draw_menu()中添加按钮绘制
    pygame.draw.rect(screen, GREEN, button_check)
    text_check = font.render('CHECK', True, BLACK)
    text_rect = text_check.get_rect(center=button_check.center)
    screen.blit(text_check, text_rect)

    
    # 绘制光标
    if pygame.time.get_ticks() % 1000 < 500:  # 闪烁效果
        if input_active_row:
            # 计算光标位置
            text_before_cursor = input_text_row[:cursor_pos_row]
            text_surface = font.render(text_before_cursor, True, BLACK)
            cursor_x = input_box_row.x + 5 + text_surface.get_width()
            pygame.draw.line(screen, BLACK, 
                           (cursor_x, input_box_row.y + 5),
                           (cursor_x, input_box_row.y + input_box_row.height - 5), 2)
        elif input_active_col:
            # 计算光标位置
            text_before_cursor = input_text_col[:cursor_pos_col]
            text_surface = font.render(text_before_cursor, True, BLACK)
            cursor_x = input_box_col.x + 5 + text_surface.get_width()
            pygame.draw.line(screen, BLACK,
                           (cursor_x, input_box_col.y + 5),
                           (cursor_x, input_box_col.y + input_box_col.height - 5), 2)
        

    # 在DRAW模式下显示绘制模式状态
    if current_mode == 'DRAW':
        mode_text = font.render(f"Mode: {'Polygon' if drawing_mode == 'polygon' else 'Curve'}", True, WHITE)
        screen.blit(mode_text, (WINDOW_WIDTH - 150, BUTTON_Y))

def draw_shapes():
    # 调整绘图区域的偏移
    drawing_offset = 30  # 为坐标轴留出空间
    
    if current_mode == 'EXPLORE':
        # 探索模式下绘制所有内容
        # 1. 绘制探索点
        for x, y, is_inside in explored_points:
            color = GREEN if is_inside else GRAY
            rect_x = ((x - COORD_MARGIN) // CELL_SIZE) * CELL_SIZE + COORD_MARGIN
            rect_y = ((y - MENU_HEIGHT) // CELL_SIZE) * CELL_SIZE + MENU_HEIGHT
            pygame.draw.rect(screen, color, (rect_x, rect_y, CELL_SIZE, CELL_SIZE))

        # 2. 绘制已完成的探索曲线
        for curve, color in explore_curves:
            if len(curve) > 1:
                pygame.draw.lines(screen, color, False, curve, 3)

        # 3. 绘制已完成的探索多边形
        for polygon in explore_polygons:
            if len(polygon) > 2:
                pygame.draw.polygon(screen, GREEN, polygon, 2)

        # 4. 绘制当前正在画的图形
        if explore_sub_mode == 'CURVE' and len(explore_curve_points) > 1:
            pygame.draw.lines(screen, explore_curve_colors[-1], False, explore_curve_points, 3)
        elif explore_sub_mode == 'POLYGON' and drawing_polygon and len(points) > 0:
            if len(points) > 1:
                pygame.draw.lines(screen, GREEN, False, points, 2)

    else:
        # DRAW 和 VERIFY 模式
        # 1. 如果是验证模式，先绘制探索点和探索线
        if current_mode == 'VERIFY':
            # 绘制探索点
            for x, y, is_inside in explored_points:
                color = GREEN if is_inside else GRAY
                rect_x = ((x - COORD_MARGIN) // CELL_SIZE) * CELL_SIZE + COORD_MARGIN
                rect_y = ((y - MENU_HEIGHT) // CELL_SIZE) * CELL_SIZE + MENU_HEIGHT
                pygame.draw.rect(screen, color, (rect_x, rect_y, CELL_SIZE, CELL_SIZE))
            
            # 绘制探索曲线
            for curve, color in explore_curves:
                if len(curve) > 1:
                    pygame.draw.lines(screen, color, False, curve, 3)
            
            # 绘制探索多边形
            for polygon in explore_polygons:
                if len(polygon) > 2:
                    pygame.draw.polygon(screen, GREEN, polygon, 2)
        
        # 2. 绘制原始多边形
        for polygon in polygons:
            pygame.draw.polygon(screen, BLACK, polygon, 2)
        
        # 3. 绘制原始曲线
        for curve in curves:
            if len(curve) > 1:
                pygame.draw.lines(screen, BLACK, False, curve, 3)
        
        # 4. 绘制当前正在画的图形
        if current_mode == 'DRAW':
            if drawing_mode == 'polygon' and drawing_polygon and len(points) > 0:
                if len(points) > 1:
                    pygame.draw.lines(screen, BLACK, False, points, 2)
                current_pos = pygame.mouse.get_pos()
                if DRAWING_AREA.collidepoint(current_pos):
                    pygame.draw.line(screen, BLACK, points[-1], current_pos, 2)
            elif drawing_mode == 'curve' and len(curve_points) > 1:
                pygame.draw.lines(screen, BLACK, False, curve_points, 3)

    # 绘制已检查的方格
    for row, col, is_inside in checked_cells:
        rect_x = col * CELL_SIZE + COORD_MARGIN
        rect_y = row * CELL_SIZE + MENU_HEIGHT
        color = GREEN if is_inside else GRAY
        pygame.draw.rect(screen, color, (rect_x, rect_y, CELL_SIZE, CELL_SIZE))

def is_in_drawing_area(pos):
    """检查点是否在绘图区域内，包括边缘"""
    x, y = pos
    return (COORD_MARGIN <= x < WINDOW_WIDTH and 
            MENU_HEIGHT <= y < WINDOW_HEIGHT)

def is_point_inside_polygon(point, polygon_points):
    x, y = point.x, point.y
    inside = False
    n = len(polygon_points)
    
    p1x, p1y = polygon_points[0]
    for i in range(n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def fill_cell_color(x, y, color):
    """填充网格颜色"""
    grid_x, grid_y = get_grid_pos((x, y))
    if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
        rect_x = grid_x * CELL_SIZE + COORD_MARGIN
        rect_y = grid_y * CELL_SIZE + MENU_HEIGHT
        pygame.draw.rect(screen, color, (rect_x, rect_y, CELL_SIZE, CELL_SIZE))

def point_in_shape(x, y, polygons, curves):
    """检查点是否在任何形状内"""
    # 转换为实际坐标
    real_x = x * CELL_SIZE + COORD_MARGIN + CELL_SIZE/2
    real_y = y * CELL_SIZE + MENU_HEIGHT + CELL_SIZE/2
    point = (real_x, real_y)
    
    # 检查多边形
    for polygon in polygons:
        if point_in_polygon(point, polygon):
            return True
            
    # 检查曲线
    for curve in curves:
        for i in range(len(curve)-1):
            if point_to_line_distance(point, curve[i], curve[i+1]) < CELL_SIZE/2:
                return True
    return False

def point_to_line_distance(point, line_start, line_end):
    """计算点到线段的距离"""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # 线段长度的平方
    l2 = (x2-x1)**2 + (y2-y1)**2
    if l2 == 0:
        return ((x-x1)**2 + (y-y1)**2)**0.5
    
    # 计算投影点的参数 t
    t = max(0, min(1, ((x-x1)*(x2-x1) + (y-y1)*(y2-y1))/l2))
    
    # 计算投影点坐标
    px = x1 + t*(x2-x1)
    py = y1 + t*(y2-y1)
    
    # 返回点到投影点的距离
    return ((x-px)**2 + (y-py)**2)**0.5

def point_near_curve(point, curve, threshold=CELL_SIZE):
    """判断点是否靠近曲线"""
    x, y = point
    for i in range(len(curve) - 1):
        x1, y1 = curve[i]
        x2, y2 = curve[i + 1]
        # 计算点到线段的距离
        dist = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / ((y2-y1)**2 + (x2-x1)**2)**0.5
        if dist < threshold:
            return True
    return False

def line_intersects_cell(line_start, line_end, cell_x, cell_y):
    """判断线段是否与网格单元相交"""
    # 计算网格单元的四个角点
    cell_left = cell_x * CELL_SIZE + COORD_MARGIN
    cell_right = cell_left + CELL_SIZE
    cell_top = cell_y * CELL_SIZE + MENU_HEIGHT
    cell_bottom = cell_top + CELL_SIZE
    
    # 线段的起点和终点
    x1, y1 = line_start
    x2, y2 = line_end
    
    # 判断线段是否与网格单元的四条边相交
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    def intersect(A, B, C, D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    # 网格单元的四条边
    cell_edges = [
        ((cell_left, cell_top), (cell_right, cell_top)),     # 上边
        ((cell_right, cell_top), (cell_right, cell_bottom)), # 右边
        ((cell_right, cell_bottom), (cell_left, cell_bottom)),# 下边
        ((cell_left, cell_bottom), (cell_left, cell_top))    # 左边
    ]
    
    # 检查线段是否与任何一条边相交
    line = (line_start, line_end)
    for edge in cell_edges:
        if intersect(line[0], line[1], edge[0], edge[1]):
            return True
            
    # 检查线段的端点是否在网格内
    def point_in_cell(point):
        x, y = point
        return (cell_left <= x <= cell_right and 
                cell_top <= y <= cell_bottom)
    
    return point_in_cell(line_start) or point_in_cell(line_end)

def check_row_col_in_shapes(row, col):
    """检查行和列对应的网格是否与任何形状相交或属于多边形内"""
    # 将网格坐标转换为实际坐标
    x = col * CELL_SIZE + COORD_MARGIN + CELL_SIZE/2
    y = row * CELL_SIZE + MENU_HEIGHT + CELL_SIZE/2
    point = (x, y)
    
    # 检查多边形
    for polygon in polygons:
        if point_in_polygon(point, polygon):
            return True
            
    # 检查曲线
    for curve in curves:
        for i in range(len(curve)-1):
            if line_intersects_cell(curve[i], curve[i+1], col, row):
                return True
    return False

def main():
    global current_mode, drawing_mode, explore_sub_mode
    global drawing_curve, curve_points, curves
    global drawing_explore_curve, explore_curve_points, explore_curves
    global drawing_polygon, points, polygons, explore_polygons
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if current_mode == 'DRAW':
                        # DRAW模式下空格键的原有逻辑保持不变
                        drawing_mode = 'polygon' if drawing_mode == 'curve' else 'curve'
                        print(f"Switched drawing mode to: {drawing_mode}")
                    
                    elif current_mode == 'EXPLORE':
                        # 探索模式下空格键循环切换子模式
                        if explore_sub_mode == EXPLORE_CLICK:
                            explore_sub_mode = EXPLORE_CURVE
                        elif explore_sub_mode == EXPLORE_CURVE:
                            explore_sub_mode = EXPLORE_POLYGON
                        else:
                            explore_sub_mode = EXPLORE_CLICK
                        print(f"Switched explore sub mode to: {explore_sub_mode}")
                        
                        # 切换模式时重置当前绘制状态
                        drawing_explore_curve = False
                        explore_curve_points = []
                        points = []
                        drawing_polygon = False

            elif event.type == pygame.MOUSEMOTION:
                if current_mode == 'EXPLORE':
                    if explore_sub_mode == EXPLORE_CURVE and drawing_explore_curve:
                        explore_curve_points.append(event.pos)

        # 绘制部分
        screen.fill(BLACK)
        draw_menu()
        draw_grid()

        if current_mode == 'DRAW':
            # 绘制已完成的曲线
            for curve in curves:
                if len(curve) > 1:
                    pygame.draw.lines(screen, WHITE, False, curve, 2)
            
            # 绘制当前正在画的曲线
            if drawing_curve and len(curve_points) > 1:
                pygame.draw.lines(screen, WHITE, False, curve_points, 2)
            
            # 绘制已完成的多边形
            for polygon in polygons:
                if len(polygon) > 2:
                    pygame.draw.polygon(screen, WHITE, polygon, 2)
            
            # 绘制当前正在画的多边形
            if drawing_polygon and len(points) > 1:
                pygame.draw.lines(screen, WHITE, False, points, 2)

        elif current_mode == 'EXPLORE':
            # 显示探索点
            for x, y, is_inside in explored_points:
                color = GREEN if is_inside else GRAY
                fill_cell_color(x, y, color)
            
            # 先绘制所有已完成的探索曲线
            for curve, color in explore_curves:
                if len(curve) > 1:
                    # 绘制曲线上的所有点
                    for point in curve:
                        pygame.draw.circle(screen, color, point, 2)
                    # 绘制点之间的连线
                    pygame.draw.lines(screen, color, False, curve, 2)
            
            if explore_sub_mode == 'CURVE':
                # 绘制当前正在画的曲线
                if drawing_explore_curve and explore_curve_points:
                    # 绘制当前曲线的所有点
                    for point in explore_curve_points:
                        pygame.draw.circle(screen, explore_curve_colors[-1], point, 2)
                    # 如果有多个点，绘制连线
                    if len(explore_curve_points) > 1:
                        pygame.draw.lines(screen, explore_curve_colors[-1], False, explore_curve_points, 2)
                    print(f"Drawing current curve with {len(explore_curve_points)} points")
            
            # 绘制所有已完成的探索多边形
            for polygon in explore_polygons:
                if len(polygon) > 2:
                    # 绘制多边形的所有顶点
                    for point in polygon:
                        pygame.draw.circle(screen, GREEN, point, 3)
                    # 绘制多边形
                    pygame.draw.polygon(screen, GREEN, polygon, 2)
            
            if explore_sub_mode == 'POLYGON':
                # 绘制当前正在画的多边形
                if points:
                    # 绘制当前多边形的所有顶点
                    for point in points:
                        pygame.draw.circle(screen, GREEN, point, 3)
                    # 如果有多个点，绘制连线
                    if len(points) > 1:
                        pygame.draw.lines(screen, GREEN, False, points, 2)

            # 显示当前子模式
            mode_text = font.render("探索模式: " + {
                'CLICK': '点击',
                'CURVE': '曲线',
                'POLYGON': '多边形'
            }[explore_sub_mode], True, WHITE)
            screen.blit(mode_text, (10, WINDOW_HEIGHT - 40))

        pygame.display.flip()
        clock.tick(60)

running = True
while running:
    screen.fill(WHITE)
    draw_menu()
    draw_grid()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.TEXTINPUT:
            # 处理文本输入
            if event.text.isdigit():  # 只允许输入数字
                if input_active_row:
                    if len(input_text_row) < max_input_length:
                        input_text_row += event.text
                        cursor_pos_row = len(input_text_row)  # 光标移动到末尾
                elif input_active_col:
                    if len(input_text_col) < max_input_length:
                        input_text_col += event.text
                        cursor_pos_col = len(input_text_col)  # 光标移动到末尾
                print(f"Current input: row={input_text_row}, col={input_text_col}")  # 打印当前输入

        # 处理键盘事件
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:  # 处理删除键
                if input_active_row:
                    if cursor_pos_row > 0:  # 如果光标不在最前面
                        input_text_row = input_text_row[:cursor_pos_row-1] + input_text_row[cursor_pos_row:]  # 删除光标前的一个字符
                        cursor_pos_row -= 1  # 光标向前移动一位
                elif input_active_col:
                    if cursor_pos_col > 0:  # 如果光标不在最前面
                        input_text_col = input_text_col[:cursor_pos_col-1] + input_text_col[cursor_pos_col:]  # 删除光标前的一个字符
                        cursor_pos_col -= 1  # 光标向前移动一位
            elif event.key == pygame.K_LEFT:  # 左键移动光标
                if input_active_row and cursor_pos_row > 0:
                    cursor_pos_row -= 1
                elif input_active_col and cursor_pos_col > 0:
                    cursor_pos_col -= 1
            elif event.key == pygame.K_RIGHT:  # 右键移动光标
                if input_active_row and cursor_pos_row < len(input_text_row):
                    cursor_pos_row += 1
                elif input_active_col and cursor_pos_col < len(input_text_col):
                    cursor_pos_col += 1
            elif event.key == pygame.K_ESCAPE and drawing_polygon:
                if len(points) >= 3:
                    polygons.append(points[:])
                points = []
                drawing_polygon = False
            # 添加模式切换快捷键（空格键）
            elif event.key == pygame.K_SPACE:
                if current_mode == 'DRAW':
                    # DRAW模式下切换曲线/多边形
                    drawing_mode = 'curve' if drawing_mode == 'polygon' else 'polygon'
                    points = []
                    curve_points = []
                    drawing_polygon = False
                    print(f"Switched drawing mode to: {drawing_mode}")
                
                elif current_mode == 'EXPLORE':
                    # 探索模式下循环切换子模式
                    if explore_sub_mode == 'CLICK':
                        explore_sub_mode = 'CURVE'
                    elif explore_sub_mode == 'CURVE':
                        explore_sub_mode = 'POLYGON'
                    else:  # POLYGON
                        explore_sub_mode = 'CLICK'
                    
                    # 切换时重置状态
                    points = []
                    explore_curve_points = []
                    drawing_explore_curve = False
                    drawing_polygon = False
                    print(f"Switched explore sub mode to: {explore_sub_mode}")
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if button_check.collidepoint(mouse_pos):
                if input_text_row and input_text_col:
                    row = int(input_text_row)
                    col = int(input_text_col)
                    # 检查行和列是否在形状内
                    is_inside = check_row_col_in_shapes(row, col)
                    # 存储检查结果
                    checked_cells.append((row, col, is_inside))
                    print(f"检查坐标: ({row}, {col}), 命中: {'是' if is_inside else '否'}")
                    # 刷新屏幕
                    pygame.display.flip()
            # 处理输入框点击
            if input_box_row.collidepoint(mouse_pos):
                input_active_row = True
                input_active_col = False
                input_selected = True  # 选中内容
            elif input_box_col.collidepoint(mouse_pos):
                input_active_col = True
                input_active_row = False
                input_selected = True  # 选中内容
            else:
                input_active_row = False
                input_active_col = False
                input_selected = False  # 取消选中
            if button_draw.collidepoint(mouse_pos):
                current_mode = 'DRAW'
                drawing_polygon = False
                points = []
            elif button_explore.collidepoint(mouse_pos):
                current_mode = 'EXPLORE'
                drawing_polygon = False
                points = []
            elif button_verify.collidepoint(mouse_pos):
                current_mode = 'VERIFY'
                drawing_polygon = False
                points = []
            elif button_reset.collidepoint(mouse_pos):
                reset_game()
            # 探索模式的处理
            elif current_mode == 'EXPLORE' and is_in_drawing_area(mouse_pos):
                if explore_sub_mode == 'CLICK':
                    # 点击探索模式的原有逻辑
                    grid_x, grid_y = get_grid_pos(mouse_pos)
                    if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                        # 检查是否在任何形状内
                        is_inside = False
                        
                        # 检查多边形
                        for polygon in polygons:
                            if point_in_polygon((mouse_pos[0], mouse_pos[1]), polygon):
                                is_inside = True
                                break
                        
                        # 如果不在多边形内，检查曲线
                        if not is_inside:
                            for curve in curves:
                                if len(curve) > 1:
                                    for i in range(len(curve)-1):
                                        if line_intersects_cell(
                                            curve[i], 
                                            curve[i+1],
                                            grid_x,
                                            grid_y
                                        ):
                                            is_inside = True
                                            break
                                if is_inside:
                                    break
                        
                        # 绘制探索结果
                        rect_x = grid_x * CELL_SIZE + COORD_MARGIN
                        rect_y = grid_y * CELL_SIZE + MENU_HEIGHT
                        color = GREEN if is_inside else GRAY
                        pygame.draw.rect(screen, color, (rect_x, rect_y, CELL_SIZE, CELL_SIZE))
                        explored_points.append((mouse_pos[0], mouse_pos[1], is_inside))
                        print(f"探索坐标: ({grid_x}, {grid_y}), 命中: {'是' if is_inside else '否'}")
                
                elif explore_sub_mode == 'CURVE':
                    # 曲线模式：开始画线
                    if is_in_drawing_area(mouse_pos):
                        if event.button == 1:  # 确保是左键点击
                            drawing_explore_curve = True
                            explore_curve_points = [mouse_pos]
                            explore_curve_colors.append(get_random_color())  # 为当前曲线生成随机颜色
                            print("Started drawing explore curve at", mouse_pos)
                
                elif explore_sub_mode == 'POLYGON':
                    # 多边形模式：处理多边形绘制
                    if pygame.mouse.get_pressed()[0]:  # 左键添加顶点
                        if not drawing_polygon:
                            drawing_polygon = True
                            points = [mouse_pos]
                        else:
                            points.append(mouse_pos)
                        print(f"Added polygon point at {mouse_pos}")
                    
                    elif pygame.mouse.get_pressed()[2]:  # 右键完成多边形
                        if drawing_polygon and len(points) >= 3:
                            explore_polygons.append(points[:])
                            print("Finished explore polygon")
                            points = []
                            drawing_polygon = False
            # 绘制模式的处理
            elif current_mode == 'DRAW' and is_in_drawing_area(mouse_pos):
                if event.button == 1:  # 左键添加点
                    if not drawing_polygon:
                        drawing_polygon = True
                        points = [mouse_pos]
                    else:
                        points.append(mouse_pos)
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if current_mode == 'DRAW' and drawing_mode == 'curve':
                if len(curve_points) > 1:
                    curves.append(curve_points[:])
                curve_points = []
            
            elif current_mode == 'EXPLORE':
                if explore_sub_mode == 'CURVE' and drawing_explore_curve:
                    if len(explore_curve_points) > 1:
                        explore_curves.append((explore_curve_points[:], explore_curve_colors[-1]))  # 存储曲线和颜色
                        print(f"Finished explore curve with {len(explore_curve_points)} points")
                    explore_curve_points = []
                    drawing_explore_curve = False
        
        elif event.type == pygame.MOUSEMOTION:
            if current_mode == 'DRAW' and drawing_mode == 'curve':
                if event.buttons[0]:  # 左键按下
                    if is_in_drawing_area(event.pos):
                        curve_points.append(event.pos)
            
            elif current_mode == 'EXPLORE':
                if explore_sub_mode == 'CURVE' and drawing_explore_curve:
                    if is_in_drawing_area(event.pos):
                        explore_curve_points.append(event.pos)

    draw_shapes()
    pygame.display.flip()

pygame.quit()
