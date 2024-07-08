from enum import IntEnum

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def draw_arrow_in_cell(grid_shape, state, action, ax):
    dx, dy = 0, 0
    arrow_length = 0.4
    arrow_offset_ratio = 1.5
    start_x_offset, start_y_offset = 0, 0
    if action == Action.UP:
        dy = arrow_length
        start_y_offset = arrow_length / arrow_offset_ratio
    elif action == Action.DOWN:
        dy = -arrow_length
        start_y_offset = -arrow_length / arrow_offset_ratio
    elif action == Action.LEFT:
        dx = -arrow_length
        start_x_offset = -arrow_length / arrow_offset_ratio
    elif action == Action.RIGHT:
        dx = arrow_length
        start_x_offset = arrow_length / arrow_offset_ratio

    ax.arrow(
        state[1] + 0.5 - start_x_offset,
        grid_shape[0] - state[0] - 0.5 - start_y_offset,
        dx,
        dy,
        head_width=0.2,
        head_length=0.2,
        fc="black",
        ec="black",
    )


def draw_terminal_state(grid_shape, terminal_state, ax):
    radius = 0.4
    circle = plt.Circle(
        (terminal_state[1] + 0.5, grid_shape[0] - terminal_state[0] - 0.5),
        radius,
        color="red",
        fill=False,
    )
    ax.add_patch(circle)
    ax.plot(
        [
            terminal_state[1] + 0.5 - radius * 0.7,
            terminal_state[1] + 0.5 + radius * 0.7,
        ],
        [
            grid_shape[0] - terminal_state[0] - 0.5 - radius * 0.7,
            grid_shape[0] - terminal_state[0] - 0.5 + radius * 0.7,
        ],
        color="red",
    )
    ax.plot(
        [
            terminal_state[1] + 0.5 - radius * 0.7,
            terminal_state[1] + 0.5 + radius * 0.7,
        ],
        [
            grid_shape[0] - terminal_state[0] - 0.5 + radius * 0.7,
            grid_shape[0] - terminal_state[0] - 0.5 - radius * 0.7,
        ],
        color="red",
    )


def draw_labyrinth_gridworld(
    grid_shape, walls, V, terminal_state, policy=None, enable_heatmap=True
):
    fig, ax = plt.subplots(figsize=(7, 7))

    if enable_heatmap:
        values = list(V.values())
        norm = mcolors.Normalize(vmin=min(values), vmax=max(values), clip=True)
        cmap = plt.get_cmap("viridis")

    for i in range(grid_shape[0] + 1):
        ax.axhline(i, color="black", lw=1)
        ax.axvline(i, color="black", lw=1)

    for state in V.keys():
        if state not in walls:
            color = "white"
            if enable_heatmap:
                color = cmap(norm(V[state]))
                ax.add_patch(
                    plt.Rectangle(
                        (state[1], grid_shape[0] - state[0] - 1), 1, 1, color=color
                    )
                )

            if policy:  # Draw policy arrows
                action = policy[state]
                draw_arrow_in_cell(grid_shape, state, action, ax)

            else:
                # Add text
                ax.text(
                    state[1] + 0.5,
                    grid_shape[0] - state[0] - 0.5,
                    f"{V[state]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if color[:3] < (0.5, 0.5, 0.5) else "black",
                )

    # Draw terminal state marking
    if terminal_state:
        draw_terminal_state(grid_shape, terminal_state, ax)

    # Draw walls
    for wall in walls:
        ax.add_patch(
            plt.Rectangle(
                (wall[1], grid_shape[0] - wall[0] - 1), 1, 1, fill=True, color="black"
            )
        )

    ax.set_xlim(0, grid_shape[1])
    ax.set_ylim(0, grid_shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().invert_yaxis()
    plt.show()


def draw_simple_gridworld(
    grid_shape,
    walls,
    V,
    terminal_states,
    policy=None,
    enable_heatmap=True,
    figsize=(7, 7),
):
    fig, ax = plt.subplots(figsize=figsize)

    if enable_heatmap:
        values = list(V.values())
        norm = mcolors.Normalize(vmin=min(values), vmax=max(values), clip=True)
        cmap = plt.get_cmap("viridis")

    for i in range(grid_shape[0] + 1):
        ax.axhline(i, color="black", lw=1)
        ax.axvline(i, color="black", lw=1)

    for row in range(grid_shape[0]):
        for col in range(grid_shape[1]):
            state = (row, col)

            if state in walls:  # Draw walls
                ax.add_patch(
                    plt.Rectangle(
                        (col, grid_shape[0] - row - 1), 1, 1, fill=True, color="black"
                    )
                )
                continue

            color = "white"
            if enable_heatmap and state in V:
                color = cmap(norm(V[state]))
                ax.add_patch(
                    plt.Rectangle((col, grid_shape[0] - row - 1), 1, 1, color=color)
                )

            if policy and state in policy:  # Draw policy arrows
                action = policy[state]
                draw_arrow_in_cell(grid_shape, state, action, ax)

            elif state in V:  # Add text for value
                ax.text(
                    col + 0.5,
                    grid_shape[0] - row - 0.5,
                    f"{V[state]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if color[:3] < (0.5, 0.5, 0.5) else "black",
                )

    # Draw terminal state markings
    for terminal_state, _ in terminal_states.items():
        draw_terminal_state(grid_shape, terminal_state, ax)

    ax.set_xlim(0, grid_shape[1])
    ax.set_ylim(0, grid_shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
