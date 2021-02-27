# Python imports.
from __future__ import print_function
from collections import defaultdict
try:
    import pygame
    import pygame.gfxdraw
    title_font = pygame.font.SysFont("CMU Serif", 48)
except ImportError:
    print("Warning: pygame not installed (needed for visuals).")

# Other imports.
import math
from simple_rl.utils.chart_utils import color_ls
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_visualizer as mdpv
from simple_rl.tasks.cookie_crumb import cookie_crumb_helpers

def _draw_state(screen,
                cookie_crumb_oomdp,
                state,
                policy=None,
                action_char_dict={},
                show_value=False,
                agent=None,
                draw_statics=True,
                agent_shape=None):
    '''
    Args:
        screen (pygame.Surface)
        cookie_crumb_oomdp (CookieCrumbOOMDP)
        state (State)
        agent_shape (pygame.rect)

    Returns:
        (pygame.Shape)
    '''

    # Make value dict.
    val_text_dict = defaultdict(lambda: defaultdict(float))
    if show_value:
        if agent is not None:
            if agent.name == 'Q-learning':
                # Use agent value estimates.
                for s in agent.q_func.keys():
                    val_text_dict[s.get_agent_x()][s.get_agent_y()] = agent.get_value(s)
            # slightly abusing the distinction between agents and planning modules...
            else:
                for s in cookie_crumb_oomdp.get_states():
                    val_text_dict[s.get_agent_x()][s.get_agent_y()] = agent.get_value(s)
        else:
            # Use Value Iteration to compute value.
            vi = ValueIteration(cookie_crumb_oomdp, sample_rate=10)
            vi.run_vi()
            for s in vi.get_states():
                val_text_dict[s.get_agent_x()][s.get_agent_y()] = vi.get_value(s)

    # Make policy dict.
    policy_dict = defaultdict(lambda : defaultdict(str))
    if policy:
        for s in cookie_crumb_oomdp.get_states():
            policy_dict[s.get_agent_x()][s.get_agent_y()] = policy(s)

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / cookie_crumb_oomdp.width
    cell_height = (scr_height - height_buffer * 2) / cookie_crumb_oomdp.height
    objects = state.get_objects()
    agent_x, agent_y = objects["agent"][0]["x"], objects["agent"][0]["y"]
    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size * 2 + 2)

    if agent_shape is not None:
        # Clear the old shape.
        pygame.draw.rect(screen, (255,255,255), agent_shape)

    # Statics
    if draw_statics:
        # Draw walls.
        for w in cookie_crumb_oomdp.walls:
            w_x, w_y = w["x"], w["y"]
            top_left_point = width_buffer + cell_width * (w_x - 1) + 5, height_buffer + cell_height * (
                    cookie_crumb_oomdp.height - w_y) + 5
            pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width - 10, cell_height - 10), 0)

    # Draw the crumbs.
    for i, c in enumerate(objects["crumb"]):
        # Dest.
        x, y = c["x"], c["y"]
        top_left_point = int(width_buffer + cell_width*(x - 1) + 70), int(height_buffer + cell_height*(cookie_crumb_oomdp.height - y) + 65)
        dest_col = (int(max(color_ls[0][0]-30, 0)), int(max(color_ls[0][1]-30, 0)), int(max(color_ls[0][2]-30, 0)))

        n, r = 6, cell_width / 8
        x, y = top_left_point[0], top_left_point[1]
        color = dest_col
        pygame.draw.polygon(screen, color, [
            (x + r * math.cos(2 * math.pi * i / n), y + r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ])

    # Draw the two goals
    for i, g in enumerate(cookie_crumb_oomdp.goals):
        dest_x, dest_y = g["x"], g["y"]
        top_left_point = int(width_buffer + cell_width*(dest_x - 1) + 37), int(height_buffer + cell_height*(cookie_crumb_oomdp.height - dest_y) + 34)
        dest_col = (int(max(color_ls[i+4][0]-30, 0)), int(max(color_ls[i+4][1]-30, 0)), int(max(color_ls[i+4][2]-30, 0)))
        pygame.draw.rect(screen, dest_col, top_left_point + (cell_width / 2, cell_height / 2))

    # Draw new agent.
    top_left_point = width_buffer + cell_width * (agent_x - 1), height_buffer + cell_height * (
                cookie_crumb_oomdp.height - agent_y)
    agent_center = int(top_left_point[0] + cell_width / 2.0), int(top_left_point[1] + cell_height / 2.0)
    agent_shape = _draw_agent(agent_center, screen, base_size=min(cell_width, cell_height) / 2.5 - 4)


    if draw_statics:
        # For each row:
        for i in range(cookie_crumb_oomdp.width):
            # For each column:
            for j in range(cookie_crumb_oomdp.height):
                top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
                r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

                # Show value of states.
                if show_value and not cookie_crumb_helpers.is_wall(cookie_crumb_oomdp, i + 1, cookie_crumb_oomdp.height - j):
                    # Draw the value.
                    val = val_text_dict[i + 1][cookie_crumb_oomdp.height - j]
                    color = mdpv.val_to_color(val)
                    pygame.draw.rect(screen, color, top_left_point + (cell_width, cell_height), 0)
                    value_text = reg_font.render(str(round(val, 2)), True, (46, 49, 49))
                    text_center_point = int(top_left_point[0] + cell_width / 2.0 - 10), int(
                        top_left_point[1] + cell_height / 3.0)
                    screen.blit(value_text, text_center_point)

                # Show optimal action to take in each grid cell.
                if policy and not cookie_crumb_helpers.is_wall(cookie_crumb_oomdp, i + 1, cookie_crumb_oomdp.height - j):
                    a = policy_dict[i+1][cookie_crumb_oomdp.height - j]
                    if a not in action_char_dict:
                        text_a = a
                    else:
                        text_a = action_char_dict[a]
                    text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/3.0)
                    text_rendered_a = cc_font.render(text_a, True, (46, 49, 49))
                    screen.blit(text_rendered_a, text_center_point)

    pygame.display.flip()

    return agent_shape

def _draw_agent(center_point, screen, base_size=30):
    '''
    Args:
        center_point (tuple): (x,y)
        screen (pygame.Surface)

    Returns:
        (pygame.rect)
    '''
    tri_bot_left = center_point[0] - base_size, center_point[1] + base_size
    tri_bot_right = center_point[0] + base_size, center_point[1] + base_size
    tri_top = center_point[0], center_point[1] - base_size
    tri = [tri_bot_left, tri_top, tri_bot_right]
    tri_color = (98, 140, 190)

    return pygame.draw.polygon(screen, tri_color, tri)
