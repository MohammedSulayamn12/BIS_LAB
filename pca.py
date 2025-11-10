import random
import time
import os

# ---------- PARAMETERS ----------
ROWS = 10
COLS = 10
p_spread = 0.4        # base probability of spread
humidity = 0.2        # dampening factor (0 = dry, 1 = no spread)
wind_direction = "E"  # can be "N", "S", "E", "W", or "NONE"
max_steps = 15        # number of simulation steps

# Initial fire or pollution sources
sources = [(5, 5)]


# ---------- NEIGHBOR FUNCTION ----------
def get_neighbors(i, j, rows, cols):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            # Check within grid
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbors.append((ni, nj))
    return neighbors


# ---------- WIND EFFECT FUNCTION ----------
def wind_favors(i, j, ni, nj):
    """Return True if wind helps spread from (ni,nj) to (i,j)."""
    if wind_direction == "N" and ni < i:  # fire spreading north
        return True
    if wind_direction == "S" and ni > i:
        return True
    if wind_direction == "E" and nj > j:
        return True
    if wind_direction == "W" and nj < j:
        return True
    return False


# ---------- DISPLAY FUNCTION ----------
def display(grid, step):
    os.system('cls' if os.name == 'nt' else 'clear')  # clear screen
    print(f"STEP {step}")
    print("Legend: 0=Safe  1=Burning/Polluted  2=Burnt/Inactive\n")
    for row in grid:
        for cell in row:
            if cell == 0:
                print(".", end=" ")  # safe
            elif cell == 1:
                print("🔥", end=" ")  # burning
            else:
                print("⬛", end=" ")  # burnt
        print()
    time.sleep(0.4)


# ---------- MAIN SIMULATION ----------
def fire_pollution_sim():
    # Initialize grid
    grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    # Set sources
    for (x, y) in sources:
        grid[x][y] = 1

    for step in range(1, max_steps + 1):
        new_grid = [row[:] for row in grid]  # copy grid for parallel update

        for i in range(ROWS):
            for j in range(COLS):
                state = grid[i][j]

                if state == 0:  # safe
                    neighbors = get_neighbors(i, j, ROWS, COLS)
                    burning_neighbors = [
                        (ni, nj) for (ni, nj) in neighbors if grid[ni][nj] == 1
                    ]
                    if not burning_neighbors:
                        continue

                    spread_prob = p_spread * (len(burning_neighbors) / len(neighbors))
                    spread_prob *= (1 - humidity)

                    # Wind effect
                    for (ni, nj) in burning_neighbors:
                        if wind_favors(i, j, ni, nj):
                            spread_prob *= 1.5
                            break

                    if random.random() < spread_prob:
                        new_grid[i][j] = 1  # catches fire/pollution

                elif state == 1:  # burning
                    new_grid[i][j] = 2  # becomes burnt

                # if 2: remains burnt (do nothing)

        grid = new_grid
        display(grid, step)

    print("\nSimulation Complete ✅")


# ---------- RUN ----------
if __name__ == "__main__":
    fire_pollution_sim()
