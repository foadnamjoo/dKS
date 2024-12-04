# %%
import numpy as np
import matplotlib.pyplot as plt
from time import time
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from math import sqrt



# %%
def generate_data(n, dimensions=2):
    """Generate n random points in [0, 1]^dimensions."""
    return np.random.uniform (0, 1, (n, dimensions))

# %%
np.random.seed(4)
n = 100
p = generate_data(n)
q = generate_data(n)

# %% [markdown]
# # Baseline O(n^2)

# %%
def baseline_alg(points1, points2):
    unique_x = np.concatenate([points1[:, 0], points2[:, 0]])
    unique_y = np.concatenate([points1[:, 1], points2[:, 1]])
    p_or_q = np.concatenate([np.ones(n), np.zeros(n)])


    max_diff = 0
    sorted_indices_y = np.argsort(unique_y);

    for i in range(len(unique_x) - 1):
        pq_diff = 0 

        #print("reset pq_diff", pq_diff)

        for j in range(len(unique_y) - 1):

            #print(i,j,sorted_indices_y[j],unique_x[sorted_indices_y[j]],unique_x[i])

            #check if x coordinate of j violates x threshold from i
            # if true, then this point is out of the x range, and move to next point in y order
            if(unique_x[i] < unique_x[sorted_indices_y[j]]): 
                continue;
            

            #print("compare:", sorted_indices_y[j], n)

            # if point j is in P:  sorted_y[j] indexes into the unique_y array, first ones in P, others in Q
            if (p_or_q[sorted_indices_y[j]] == 1) :
                pq_diff += 1
            else:
                pq_diff -= 1 

            #print(sorted_indices_y[j], pq_diff)

            if abs(pq_diff) > max_diff:
                max_diff = abs(pq_diff)

    # print(max_diff)

    float_diff = float(max_diff)
    float_n = float(n)

    return float_diff / float_n


baseline_alg(p, q)



# %% [markdown]
# # Optimized Algorithm 

# %%
def optimized_function(p, q):
    n = p.shape[0]
    grid_size = 2 * int(sqrt(n))
    
    # Initialize counters to each cell to zero, a
    counts_p = np.zeros((grid_size, grid_size), dtype=int)
    counts_q = np.zeros((grid_size, grid_size), dtype=int)

    # Step 1: Sort x- and y-coordinates for both p and q
    sorted_x_p = np.sort(p[:, 0])
    sorted_y_p = np.sort(p[:, 1])
    sorted_x_q = np.sort(q[:, 0])
    sorted_y_q = np.sort(q[:, 1])

    # Step 2: Merge and sort the coordinates from both p and q
    combined_sorted_x = np.sort(np.concatenate([sorted_x_p, sorted_x_q]))
    combined_sorted_y = np.sort(np.concatenate([sorted_y_p, sorted_y_q]))



    # Step 3: Select evenly spaced grid boundaries based on the combined sorted coordinates
    # Choose grid_size + 1 points to include boundaries
    selected_x = combined_sorted_x[::len(combined_sorted_x) // (grid_size + 1)]
    selected_y = combined_sorted_y[::len(combined_sorted_y) // (grid_size + 1)]


    # Assign indices to points
    def assign_indices(points, selected_x, selected_y):
        indices = []
        for x, y in points:
            i = np.searchsorted(selected_x, x, side='right') - 1
            j = np.searchsorted(selected_y, y, side='right') - 1
            indices.append((min(i, grid_size - 1), min(j, grid_size - 1)))  # Ensure indices are within bounds
        return indices

    indices_p = assign_indices(p, selected_x, selected_y)
    indices_q = assign_indices(q, selected_x, selected_y)


    # Increment counters based on indices
    for i, j in indices_p:
        counts_p[i, j] += 1
    for i, j in indices_q:
        counts_q[i, j] += 1


    # Compute cumulative sum column-wise
    Yp = np.cumsum(counts_p, axis=0)
    Yq = np.cumsum(counts_q, axis=0)


    def cumulative_count(Yx):
        rows, cols = grid_size, grid_size

        # Initialize Cx with zeros (or an initial value)
        Cx = np.zeros_like(Yx)
        # print(Cx.shape)
        # print(Yx.shape)

        # Compute Cx based on the formula
        for i in range(rows):
            for j in range(cols):
                if i == 0:  # Special case for the first row
                    Cx[i, j] = Yx[i, j-1] if j > 0 else 0
                  # Cx[i, j] = 0
                elif j == 0:  # Special case for the first column
                    Cx[i, j] = Cx[i-1, j]
                   # Cx[i, j] =  0
                    
                else:
                  # Cx[i, j] = Cx[i-1, j] + Yx[i, j-1]
                    Cx[i, j] = Cx[i-1, j] + Cx[i, j-1] - Cx[i-1, j-1] + Yx[i, j]

        return Cx
    



    Cp = cumulative_count(Yp)
    Cq = cumulative_count(Yq)


#from here checked !

    difference = np.abs(Cp - Cq)
    max_diff = np.max(difference)
    # max_diff_position = np.unravel_index(np.argmax(difference), difference.shape)


 # Plot the heatmap
    # plt.figure(figsize=(8, 6))
    # plt.imshow(difference, cmap='hot', interpolation='nearest')
    # plt.colorbar(label='|Cp(i, j) - Cq(i, j)|')
    # plt.title('Heatmap of Differences |Cp - Cq|')
    # plt.xlabel('j (Columns)')
    # plt.ylabel('i (Rows)')
    # plt.show()

    # print(f"Maximum difference: {max_diff}")
    # print(f"Position of maximum difference: {max_diff_position}")
    # print("Grid Size:", grid_size)
    # print("indices_p", indices_p)
    # print("indices_q", indices_q)
    # print("Differences", difference)

    float_diff = float(max_diff)
    float_n = float(n)

    return float_diff / float_n

optimized_function(p, q)

# %%
# Cp, Cq, max_diff, max_diff_position = optimized_function(p, q)

# print("Cp:")
# print(Cp)
# print("\nCq:")
# print(Cq)
# print("\nMaximum absolute difference:")
# print(max_diff/n)
# print("max diff position", max_diff_position)


# %% [markdown]
# # Evaluation

# %%
# Generate datasets and evaluate algorithms
n_values = np.logspace(1, 4, num=10, dtype=int)
baseline_runtimes = []
optimized_runtimes = []
baseline_errors = []
optimized_errors = []

for n in n_values:
    p = generate_data(n)
    q = generate_data(n)
    k = int(2 * np.sqrt(n))
    
    # Baseline algorithm
    start_time = time()
    baseline_error = baseline_alg(p, q)
    baseline_runtimes.append(time() - start_time)
    baseline_errors.append(baseline_error)
    
    # Optimized algorithm
    start_time = time()
    optimized_error = optimized_function(p, q)
    optimized_runtimes.append(time() - start_time)
    optimized_errors.append(optimized_error)

# Plot results
plt.figure(figsize=(18, 5))

# Plot 1: Runtime vs n
plt.subplot(1, 3, 1)
plt.plot(n_values, baseline_runtimes, label='Baseline')
plt.plot(n_values, optimized_runtimes, label='Optimized')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n (# samples)')
plt.ylabel('Runtime (s)')
plt.title('Runtime vs n')
plt.legend()

# Plot 2: Error vs n
plt.subplot(1, 3, 2)
plt.plot(n_values, baseline_errors, label='Baseline')
plt.plot(n_values, optimized_errors, label='Optimized')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n (# samples)')
plt.ylabel('Maximum Difference / n')
plt.title('Error vs n')
plt.legend()

# Plot 3: Error vs Runtime
plt.subplot(1, 3, 3)
plt.plot(baseline_runtimes, baseline_errors, label='Baseline')
plt.plot(optimized_runtimes, optimized_errors, label='Optimized')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Runtime (s)')
plt.ylabel('Maximum Difference / n')
plt.title('Error vs Runtime')
plt.legend()

plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from time import time



# Baseline Algorithm
def process_points(points1, points2, step):
    combined_x = np.unique(np.concatenate([points1[:, 0], points2[:, 0]]))
    sampled_x = combined_x[::step]
    if sampled_x[-1] < combined_x[-1]:
        sampled_x = np.append(sampled_x, combined_x[-1])
    
    combined_y = np.unique(np.concatenate([points1[:, 1], points2[:, 1]]))
    sampled_y = combined_y[::step]
    if sampled_y[-1] < combined_y[-1]:
        sampled_y = np.append(sampled_y, combined_y[-1])

    differences = []
    for i in range(len(sampled_x) - 1):
        for j in range(len(sampled_y) - 1):
            x_min, x_max = sampled_x[i], sampled_x[i + 1]
            y_min, y_max = sampled_y[j], sampled_y[j + 1]

            count1 = np.sum((x_min <= points1[:, 0]) & (points1[:, 0] <= x_max) &
                            (y_min <= points1[:, 1]) & (points1[:, 1] <= y_max))
            count2 = np.sum((x_min <= points2[:, 0]) & (points2[:, 0] <= x_max) &
                            (y_min <= points2[:, 1]) & (points2[:, 1] <= y_max))
            differences.append(abs((count1 / points1.shape[0]) - (count2 / points2.shape[0])))

    return max(differences)

# Optimized Algorithm
def optimized_function(points1, points2):
    n = points1.shape[0]
    grid_size = int(np.sqrt(n))
    counts_p = np.zeros((grid_size, grid_size), dtype=int)
    counts_q = np.zeros((grid_size, grid_size), dtype=int)

    sorted_x = np.sort(np.concatenate([points1[:, 0], points2[:, 0]]))
    sorted_y = np.sort(np.concatenate([points1[:, 1], points2[:, 1]]))

    boundaries_x = sorted_x[::len(sorted_x) // (grid_size + 1)]
    boundaries_y = sorted_y[::len(sorted_y) // (grid_size + 1)]

    def assign_to_grid(points, boundaries_x, boundaries_y):
        indices = []
        for x, y in points:
            i = np.searchsorted(boundaries_x, x, side='right') - 1
            j = np.searchsorted(boundaries_y, y, side='right') - 1
            indices.append((min(i, grid_size - 1), min(j, grid_size - 1)))
        return indices

    indices_p = assign_to_grid(points1, boundaries_x, boundaries_y)
    indices_q = assign_to_grid(points2, boundaries_x, boundaries_y)

    for i, j in indices_p:
        counts_p[i, j] += 1
    for i, j in indices_q:
        counts_q[i, j] += 1

    cp = np.cumsum(np.cumsum(counts_p, axis=0), axis=1)
    cq = np.cumsum(np.cumsum(counts_q, axis=0), axis=1)

    max_diff = np.max(np.abs(cp - cq) / n)
    return max_diff

# Measure runtime
def measure_runtime(func, *args):
    start_time = time()
    result = func(*args)
    end_time = time()
    return result, end_time - start_time

# Compute error
def compute_error(alg_result, ground_truth):
    return abs(alg_result - ground_truth)

# Main experiment
def experiment():
    sample_sizes = [64, 128]
    baseline_runtimes = []
    optimized_runtimes = []
    baseline_errors = []
    optimized_errors = []

    # Generate ground truth using a large sample size
    ground_truth_size = 128
    p_gt = generate_data(ground_truth_size)
    q_gt = generate_data(ground_truth_size)
    ground_truth = process_points(p_gt, q_gt, step=2)

    for n in sample_sizes:
        p = generate_data(n)
        q = generate_data(n)

        # Baseline
        baseline_result, baseline_time = measure_runtime(process_points, p, q)
        baseline_error = compute_error(baseline_result, ground_truth)

        # Optimized
        optimized_result, optimized_time = measure_runtime(optimized_function, p, q)
        optimized_error = compute_error(optimized_result, ground_truth)

        baseline_runtimes.append(baseline_time)
        optimized_runtimes.append(optimized_time)
        baseline_errors.append(baseline_error)
        optimized_errors.append(optimized_error)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, baseline_runtimes, label="Baseline Runtime", marker="o")
    plt.plot(sample_sizes, optimized_runtimes, label="Optimized Runtime", marker="o")
    plt.xlabel("Sample Size (n)")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.title("Runtime vs. Sample Size")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, baseline_errors, label="Baseline Error", marker="o")
    plt.plot(sample_sizes, optimized_errors, label="Optimized Error", marker="o")
    plt.xlabel("Sample Size (n)")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Error vs. Sample Size")
    plt.show()

    # Combined Plot
    plt.figure(figsize=(10, 6))
    plt.plot([1 / e for e in baseline_errors], baseline_runtimes, label="Baseline", marker="o")
    plt.plot([1 / e for e in optimized_errors], optimized_runtimes, label="Optimized", marker="o")
    plt.xlabel("1/Error")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.title("Runtime vs. 1/Error")
    plt.show()

# Run the experiment
experiment()


# %% [markdown]
# # Jeff Approach

# %%
def process_points(points1, points2, step):
    # Combine, deduplicate, sort, and downsample first column
    combined_x = np.unique(np.concatenate([points1[:, 0], points2[:, 0]]))
    sampled_x = combined_x[::step]

    # Ensure the right-most boundary is included
    if sampled_x[-1] < combined_x[-1]:
        sampled_x = np.append(sampled_x, combined_x[-1])

    # Combine, deduplicate, sort, and downsample second column
    combined_y = np.unique(np.concatenate([points1[:, 1], points2[:, 1]]))
    sampled_y = combined_y[::step]

    # Ensure the upper-most boundary is included
    if sampled_y[-1] < combined_y[-1]:
        sampled_y = np.append(sampled_y, combined_y[-1])

    # Initialize result
    differences = []

    # Iterate through grid cells (including right-most and upper-most)
    for i in range(len(sampled_x) - 1):
        for j in range(len(sampled_y) - 1):
            x_min, x_max = sampled_x[i], sampled_x[i + 1]
            y_min, y_max = sampled_y[j], sampled_y[j + 1]

            # Count points in this grid cell
            count1 = np.sum(
                (x_min <= points1[:, 0]) & (points1[:, 0] <= x_max) &
                (y_min <= points1[:, 1]) & (points1[:, 1] <= y_max)
            )
            count2 = np.sum(
                (x_min <= points2[:, 0]) & (points2[:, 0] <= x_max) &
                (y_min <= points2[:, 1]) & (points2[:, 1] <= y_max)
            )

            # Calculate difference
            differences.append(abs((count1 / points1.shape[0]) - (count2 / points2.shape[0])))

    return sampled_x, sampled_y, differences



# %%
epsilon = 0.1
step = int(2/epsilon)
process_points(p, q, step)

# %%
def plot_grid_and_points(points1, points2, sampled_x, sampled_y):
    # Plot the points
    plt.figure(figsize=(10, 10))
    plt.scatter(points1[:, 0], points1[:, 1], color='blue', alpha=0.6, label='Set 1')
    plt.scatter(points2[:, 0], points2[:, 1], color='red', alpha=0.6, label='Set 2')

    # Plot the grid
    for x in sampled_x:
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.7)
    for y in sampled_y:
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.7)

    # Labels and legend
    plt.title('Point Sets and Grid Boundaries')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(False)
    plt.show()



sampled_x, sampled_y, differences = process_points(p, q, step)

# Plot the points and grid
plot_grid_and_points(p, q, sampled_x, sampled_y)
print(len(differences))
print(f"Step value: {step}")


# %%
def plot_grid_with_color(points1, points2, sampled_x, sampled_y, differences, max_diff, runtime):
    # Reshape differences to fit the grid
    differences = np.array(differences)
    grid_differences = differences.reshape(len(sampled_x) - 1, len(sampled_y) - 1)

    # Normalize differences for color intensity
    norm = plt.Normalize(vmin=np.min(differences), vmax=np.max(differences))
    cmap = plt.cm.hot # Use a single-hue colormap (Blues)

    # Plot the points
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points1[:, 0], points1[:, 1], color='blue', alpha=0.6, label='Set 1')
    ax.scatter(points2[:, 0], points2[:, 1], color='red', alpha=0.6, label='Set 2')

    # Color the grid cells
    for i in range(len(sampled_x) - 1):
        for j in range(len(sampled_y) - 1):
            x_min, x_max = sampled_x[i], sampled_x[i + 1]
            y_min, y_max = sampled_y[j], sampled_y[j + 1]
            diff = grid_differences[i, j]

            # Create a rectangle patch and add it to the plot
            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                color=cmap(norm(diff)),
                alpha=0.7
            )
            ax.add_patch(rect)

    # Add colorbar explicitly associated with the ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # ScalarMappable for colorbar
    sm.set_array([])  # Set data for the colorbar
    plt.colorbar(sm, ax=ax, label='Difference Intensity')

    # Add maximum difference and runtime as text
    ax.text(1.05, 0.95, f'Max Difference: {max_diff:.4f}', transform=ax.transAxes, fontsize=12, color='black')
    ax.text(1.05, 0.90, f'Runtime: {runtime:.2f}s', transform=ax.transAxes, fontsize=12, color='black')

    # Labels, legend, and grid
    ax.set_title('Grid Differences with Color Intensity')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()
    ax.grid(False)

    plt.show()

# Example usage
# Process the points and get grid differences
sampled_x, sampled_y, differences = process_points(p, q, step)

# Calculate maximum difference and runtime for example
start_time = time()
max_diff = max(differences)
runtime = time() - start_time

# Plot the grid with colored differences and annotations
plot_grid_with_color(p, q, sampled_x, sampled_y, differences, max_diff, runtime)


# %%
import cProfile
import pstats
import io
from pstats import SortKey

def profile_optimized_algorithm():
    # Generate small datasets
    points1 = generate_data(100)
    points2 = generate_data(100)

    # Profile the optimized algorithm
    profiler = cProfile.Profile()
    profiler.enable()
    optimized_function(points1, points2)  # Call the optimized algorithm
    profiler.disable()

    # Output the profiling results
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(SortKey.TIME)
    stats.print_stats()

    # Print results
    print(stream.getvalue())

# Run the profiler
profile_optimized_algorithm()


# %%
import timeit

# Wrapper functions for timing parts of the optimized algorithm
def time_grid_initialization(points1, points2):
    grid_size = int(np.sqrt(points1.shape[0]))
    counts_p = np.zeros((grid_size, grid_size), dtype=int)
    counts_q = np.zeros((grid_size, grid_size), dtype=int)

def time_sorting(points1, points2):
    sorted_x_p = np.sort(points1[:, 0])
    sorted_y_p = np.sort(points1[:, 1])
    sorted_x_q = np.sort(points2[:, 0])
    sorted_y_q = np.sort(points2[:, 1])

def time_cumulative_count(points1, points2):
    grid_size = int(np.sqrt(points1.shape[0]))
    counts_p = np.zeros((grid_size, grid_size), dtype=int)
    Yp = np.cumsum(counts_p, axis=0)

# Generate small datasets
points1 = generate_data(1000000)
points2 = generate_data(1000000)

# Measure time for each section
grid_init_time = timeit.timeit(lambda: time_grid_initialization(points1, points2), number=100)
sorting_time = timeit.timeit(lambda: time_sorting(points1, points2), number=100)
cumulative_count_time = timeit.timeit(lambda: time_cumulative_count(points1, points2), number=100)

# Print results
print(f"Grid Initialization Time: {grid_init_time:.6f} seconds")
print(f"Sorting Time: {sorting_time:.6f} seconds")
print(f"Cumulative Count Time: {cumulative_count_time:.6f} seconds")


# %%
# %% [markdown]
# # Runtime vs. Sample Size Comparison

# %%
def measure_runtime(func, *args):
    """Measure the runtime of a function."""
    start_time = time()
    func(*args)
    return time() - start_time

# Sample sizes to test
sample_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16400]

# Store runtime results
baseline_runtimes = []
optimized_runtimes = []

epsilon = 0.1  # Adjust epsilon for baseline
step = int(2 / epsilon)

for n in sample_sizes:
    points1 = generate_data(n)
    points2 = generate_data(n)
    
    # Measure baseline runtime
    baseline_runtime = measure_runtime(process_points, points1, points2, step)
    baseline_runtimes.append(baseline_runtime)
    
    # Measure optimized runtime
    optimized_runtime = measure_runtime(optimized_function, points1, points2)
    optimized_runtimes.append(optimized_runtime)

# Plot Runtime vs. Sample Size
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, baseline_runtimes, label='Baseline Algorithm (process_points)', marker='o', linestyle='-')
plt.plot(sample_sizes, optimized_runtimes, label='Optimized Algorithm (optimized_function)', marker='s', linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Sample Size (n)')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime vs. Sample Size')
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.show()


# %% [markdown]
# ## example of making Cp

# %%
import numpy as np

# Example sizes for Yp and Cp
rows, cols = 5, 5

# Initialize Yp with some example values
Yp = np.random.randint(1, 10, (rows, cols))

# Initialize Cp with zeros (or an initial value)
Cp = np.zeros_like(Yp)

# Compute Cp based on the formula
for i in range(rows):
    for j in range(cols):
        if i == 0:  # Special case for the first row
            Cp[i, j] = Yp[i, j-1] if j > 0 else 0
        elif j == 0:  # Special case for the first column
            Cp[i, j] = Cp[i-1, j]
        else:
            Cp[i, j] = Cp[i-1, j] + Yp[i, j-1]

# Print the results
print("Yp (input):")
print(Yp)
print(Yp.shape)

print("\nCp (calculated):")
print(Cp)
print(Cp.shape)




# %% [markdown]
# # Evaluation

# %% [markdown]
# # Optimized Algorithm with sqrt{n}

# %%
def evenly_spaced_indices(sorted_list, sqrt_n):
    """Select evenly spaced indices from a sorted list."""
    step = len(sorted_list) // sqrt_n
    return sorted_list[::step][:sqrt_n]

# %%
sorted_list = np.sort(p) 
sorted_list = np.sort(q) 

 # A sorted list of 20 evenly spaced numbers
print("srtd lst:", sorted_list)
sqrt_n = 5  # Number of intervals or evenly spaced points to select


# %%
evenly_spaced_points = evenly_spaced_indices(sorted_list, sqrt_n)
print("#intrvals:", sqrt_n)
print("Evenly Spcd:", evenly_spaced_points)

# %%
def assign_indices(p, q, selected_x, selected_y):
    """Assign each point to grid indices based on selected evenly spaced points."""
    x_indices = np.searchsorted(selected_x, p[:, 0], side='right') - 1
    y_indices = np.searchsorted(selected_y, q[:, 1], side='right') - 1
    return np.column_stack((x_indices, y_indices))

# %%


# Call the function
indices = assign_indices(p, q, selected_x, selected_y)

# Print inputs and results
#print("Points in P:", p)
#print("Points in Q:", q)
print("Selected X boundaries:", selected_x)
print("Selected Y boundaries:", selected_y)
print("Assigned Indices (Grid Cells):", indices)


# %%
def cumulative_totals(array):
    """Compute cumulative totals for the array using dynamic programming."""
    return np.cumsum(np.cumsum(array, axis=0), axis=1)

def normalize_array(array):
    """Normalize an array to the range [0, 1]."""
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def plot_heatmap(array, title, x_ticks, y_ticks):
    """Plot a heatmap of a 2D array with axis ticks."""
    plt.figure(figsize=(8, 6))
    plt.imshow(array, cmap='hot', interpolation='nearest', origin='upper')
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xticks(ticks=np.arange(len(x_ticks)), labels=np.round(x_ticks, 2), rotation=45)
    plt.yticks(ticks=np.arange(len(y_ticks)), labels=np.round(y_ticks, 2))
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.tight_layout()
    plt.show()


# Compute parameters
n = p.shape[0]
sqrt_n = int(np.sqrt(n))

# Sort points
points_sorted_x_p = np.sort(p[:, 0])
points_sorted_y_p = np.sort(p[:, 1])
points_sorted_x_q = np.sort(q[:, 0])
points_sorted_y_q = np.sort(q[:, 1])

# Select sqrt{n} evenly spaced points
selected_x_p = evenly_spaced_indices(points_sorted_x_p, sqrt_n)
selected_y_p = evenly_spaced_indices(points_sorted_y_p, sqrt_n)
selected_x_q = evenly_spaced_indices(points_sorted_x_q, sqrt_n)
selected_y_q = evenly_spaced_indices(points_sorted_y_q, sqrt_n)

# Assign indices and initialize counters
P = np.zeros((sqrt_n, sqrt_n), dtype=int)
Q = np.zeros((sqrt_n, sqrt_n), dtype=int)

indices_p = assign_indices(p, selected_x_p, selected_y_p)
indices_q = assign_indices(q, selected_x_q, selected_y_q)

for i, j in indices_p:
    P[i, j] += 1
for i, j in indices_q:
    Q[i, j] += 1

# Compute cumulative totals
CP = cumulative_totals(P)
CQ = cumulative_totals(Q)

# Compute normalized absolute difference
difference = np.abs(CP - CQ)
normalized_difference = normalize_array(difference)

# Find maximum difference
max_difference = np.max(difference)
print("Maximum Difference:", normalized_difference)

# Plot normalized heatmap
plot_heatmap(normalized_difference, "Normalized Absolute Difference |CP - CQ|", selected_x_p, selected_y_p)


# %% [markdown]
# # Optimized Algorithm with Epsilom

# %%
from time import time

def optimized_algorithm(points1, points2, epsilon):
    step = int(2 / epsilon)

    # Sort points by axes
    points1_sorted_x = np.sort(points1[:, 0])
    points1_sorted_y = np.sort(points1[:, 1])
    points2_sorted_x = np.sort(points2[:, 0])
    points2_sorted_y = np.sort(points2[:, 1])

    # Define grid boundaries
    grid_x = np.sort(np.unique(np.concatenate([
        points1_sorted_x[::step],
        points2_sorted_x[::step]
    ])))
    grid_y = np.sort(np.unique(np.concatenate([
        points1_sorted_y[::step],
        points2_sorted_y[::step]
    ])))

    # Dynamic programming setup
    counts_p = np.zeros((len(grid_x), len(grid_y)))
    counts_q = np.zeros((len(grid_x), len(grid_y)))

    # Increment counts for each grid cell
    for px, py in points1:
        x_idx = np.searchsorted(grid_x, px, side='right') - 1
        y_idx = np.searchsorted(grid_y, py, side='right') - 1
        counts_p[x_idx, y_idx] += 1

    for qx, qy in points2:
        x_idx = np.searchsorted(grid_x, qx, side='right') - 1
        y_idx = np.searchsorted(grid_y, qy, side='right') - 1
        counts_q[x_idx, y_idx] += 1

    # Cumulative counts for efficient queries
    cum_counts_p = np.cumsum(np.cumsum(counts_p, axis=0), axis=1)
    cum_counts_q = np.cumsum(np.cumsum(counts_q, axis=0), axis=1)

    # Compute maximum difference
    differences = np.abs(cum_counts_p - cum_counts_q) / len(points1)
    max_difference = np.max(differences)

    return max_difference, differences, grid_x, grid_y

def plot_differences(grid_x, grid_y, differences):
    plt.figure(figsize=(10, 10))
    plt.imshow(differences.T, extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]), origin='lower', cmap='coolwarm')
    plt.colorbar(label='Difference')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Grid Differences')
    plt.show()


start_time = time()
max_diff, diffs, grid_x, grid_y = optimized_algorithm(p, q, epsilon)
runtime = time() - start_time

print(f"Maximum Difference: {max_diff}")
print(f"Runtime: {runtime:.2f} seconds")

# Plot Differences
plot_differences(grid_x, grid_y, diffs)


# %% [markdown]
# # Second Approach

# %%

def create_grid(data, epsilon):
    """Create a grid for ranges using a net with 2/epsilon points."""
    # Sort data by first column, then second column
    data = data[np.lexsort((data[:, 1], data[:, 0]))]

    axis_points = int(2 / epsilon)
    n = len(data)

    # Get sorted x and y values
    sorted_x = data[:, 0]
    sorted_y = data[:, 1]

    # Select grid boundaries based on epsilon
    x_boundaries = sorted_x[np.linspace(0, n - 1, axis_points, dtype=int)]

    y_boundaries = sorted_y[np.linspace(0, n - 1, axis_points, dtype=int)]
    
    print("grid is created.")
    return x_boundaries, y_boundaries



def plot_grid(data, x_boundaries, y_boundaries):
    """Plot the grid and data points."""
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], s=10, label="Data Points", color='blue')

    # Draw vertical grid lines
    for x in x_boundaries:
        plt.axvline(x=x, color='red', linestyle='--', linewidth=0.7)
    
    # Draw horizontal grid lines
    for y in y_boundaries:
        plt.axhline(y=y, color='green', linestyle='--', linewidth=0.7)

    # Add labels and legend
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Grid Visualization with Data Points")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate random data points
    np.random.seed(42)
    data = np.random.rand(100, 2)  # 100 data points in 2D space

    epsilon = 0.2
    x_boundaries, y_boundaries = create_grid(data, epsilon)
    plot_grid(data, x_boundaries, y_boundaries)

# %%


def create_approximate_grid(data, epsilon, d):
    """
    Create an approximate grid for multi-dimensional Kolmogorov-Smirnov distance.

    Args:
        data (ndarray): Input data points (n x d).
        epsilon (float): Approximation parameter.
        d (int): Dimensionality of the data.

    Returns:
        list: List of grid boundaries for each dimension.
    """
    # Calculate the number of grid divisions
    num_divisions = int(d / epsilon)
    
    # Sort data along each dimension
    sorted_data = [np.sort(data[:, i]) for i in range(d)]
    
    # Choose grid boundaries using stratified sampling
    grid_boundaries = [
        sorted_data[i][np.linspace(0, len(data) - 1, num_divisions, dtype=int)]
        for i in range(d)
    ]
    
    return grid_boundaries

def main():
    # Generate random data for demonstration
    np.random.seed(42)  # For reproducibility
    data_points = 100
    data = np.random.rand(data_points, 2)  # 100 random points in a 2D space

    epsilon = 0.1
    d = 2  # 2D data

    # Create the grid
    grid_boundaries = create_approximate_grid(data, epsilon, d)

    # Plot the grid and data points
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], label='Data Points', alpha=0.7)

    # Draw grid lines
    for x in grid_boundaries[0]:
        plt.axvline(x=x, color='blue', linestyle='--', linewidth=0.5)
    for y in grid_boundaries[1]:
        plt.axhline(y=y, color='red', linestyle='--', linewidth=0.5)

    plt.title('Approximate Grid for Multi-Dimensional KS Distance')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()


# %%


# %%


# %%
data = generate_data(100)

#data

# %%
def baseline_dks(n_point, epsilon):

    array1 = generate_data(n_point)  # Random 2D points for demonstration
    array2 = generate_data(n_point)  # Another set of random 2D points

    X1, Y1 = create_grid(array1, epsilon)
    X2, Y2 = create_grid(array2, epsilon)

    # Create the grid
    grid_x = np.linspace(min(X1.min(), X2.min()), max(X1.max(), X2.max()), len(X1) * 2)
    grid_y = np.linspace(min(Y1.min(), Y2.min()), max(Y1.max(), Y2.max()), len(Y1) * 2)

    # Calculate 2D histograms
    hist1, _, _ = np.histogram2d(array1[:, 0], array1[:, 1], bins=[grid_x, grid_y])
    hist2, _, _ = np.histogram2d(array2[:, 0], array2[:, 1], bins=[grid_x, grid_y])

    # Compute the difference in counts
    difference = abs(hist1 / len(array1) - hist2 / len(array2))

    # Output the result
    #print("Difference between the two histograms:\n", difference)
    print(difference.max())
    print(difference.shape)

    return(difference)


# %%
epsilon = 0.1
sample_sizes = [12800, 25600, 51200, 102400, 204800]
baseline_time_list = []
for n in sample_sizes:
    start = time.time()
    difference = baseline_dks(n, epsilon)
    baseline_time = time.time() - start
    baseline_time_list.append(baseline_time)


# %%
plt.plot(sample_sizes, baseline_time_list)

# %% [markdown]
# # Previous Work

# %%
create_grid(data, 0.1)

# %%
#data[np.lexsort((data[:, 1], data[:, 0]))]

# %%
x_b, y_b = create_grid(data, 2)
print(x_b, y_b)
x_b
y_b

# %%
plt.scatter(data[:, 0], data[:, 1])
# Horizontal line at y=0.5
plt.axhline(y=y_b, color='blue', linestyle='--', linewidth=2, label='Horizontal Line (y=0.5)')
# Vertical line at x=0.3
plt.axvline(x=x_b, color='red', linestyle=':', linewidth=2, label='Vertical Line (x=0.3)')


# %%
[None] * 6

# %%

def count_in_ranges(data, x_bounds, y_bounds):
    """Count points in each grid cell."""
    grid_counts = np.zeros((len(x_bounds), len(y_bounds)))

    for point in data:
        x_index = np.searchsorted(x_bounds, point[0], side='right') - 1
        y_index = np.searchsorted(y_bounds, point[1], side='right') - 1
        if 0 <= x_index < len(x_bounds) and 0 <= y_index < len(y_bounds):
            grid_counts[x_index, y_index] += 1

    return grid_counts


# %%
count_in_ranges(data, x_b, y_b)

# %%
def dks_distance(P, Q, epsilon):
    """Compute dKS distance using optimized grid-based approach."""
    x_bounds, y_bounds = create_grid(P, epsilon)
    grid_P = count_in_ranges(P, x_bounds, y_bounds)
    grid_Q = count_in_ranges(Q, x_bounds, y_bounds)
    
    # Compute maximum normalized difference
    n = len(P)  # Assume |P| = |Q|
    differences = np.abs(grid_P - grid_Q) / n
    return np.max(differences)

# %%
def baseline_dks(P, Q):
    """Baseline dKS distance by comparing all pairs of points."""
    n = len(P)
    max_difference = 0

    for i in range(n):
        for j in range(n):
            range_P = (P[:, 0] <= P[i, 0]) & (P[:, 1] <= P[j, 1])
            range_Q = (Q[:, 0] <= P[i, 0]) & (Q[:, 1] <= P[j, 1])
            difference = abs(range_P.sum() - range_Q.sum()) / n
            max_difference = max(max_difference, difference)

    return max_difference

# %%
def evaluate(n, epsilon):
    """Evaluate runtime, accuracy, and performance."""
    P = generate_data(n)
    Q = generate_data(n)
    print(P.shape, Q.shape)

    # Optimized algorithm
    start = time.time()
    dks_optimized = dks_distance(P, Q, epsilon)
    optimized_time = time.time() - start

    # Baseline algorithm
    start = time.time()
    dks_baseline = baseline_dks(P, Q)
    baseline_time = time.time() - start

    # Calculate relative error
    relative_error = abs(dks_baseline - dks_optimized) / dks_baseline

    return {
        "Optimized dKS": float(dks_optimized),
        "Baseline dKS": float(dks_baseline),
        "Optimized Time (s)": float(optimized_time),
        "Baseline Time (s)": float(baseline_time),
        "Relative Error": float(relative_error)
    }

# %%
def run_experiments(sample_sizes, epsilon):
    """Run experiments for multiple sample sizes and return results."""
    results = []
    for n in sample_sizes:
        res = evaluate(n, epsilon)
        res["Sample Size"] = n
        results.append(res)
        print(f"Completed for n={n}: {res}")
    return results

# %%
def plot_results(results):
    """Generate runtime and error plots from results with colorblind-friendly adjustments."""
    sample_sizes = [res["Sample Size"] for res in results]
    optimized_times = [res["Optimized Time (s)"] for res in results]
    baseline_times = [res["Baseline Time (s)"] for res in results]
    relative_errors = [res["Relative Error"] for res in results]

    # Runtime vs. Sample Size
    plt.figure()
    plt.plot(sample_sizes, optimized_times, label="Optimized Algorithm", color="C0", linestyle="-", marker="o")
    plt.plot(sample_sizes, baseline_times, label="Baseline Algorithm", color="C1", linestyle="--", marker="s")
    plt.xlabel("Sample Size")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs. Sample Size")
    plt.legend()
    plt.grid()

    # Error vs. Sample Size
    plt.figure()
    plt.plot(sample_sizes, relative_errors, label="Relative Error", color="C2", linestyle=":", marker="^")
    plt.xlabel("Sample Size")
    plt.ylabel("Relative Error")
    plt.title("Error vs. Sample Size")
    plt.legend()
    plt.grid()

    # Runtime vs. 1/Error
    plt.figure()
    inverse_errors = [1 / err if err > 0 else 0 for err in relative_errors]
    plt.plot(inverse_errors, optimized_times, label="Optimized Algorithm", color="C3", linestyle="-.", marker="d")
    plt.xlabel("1/Error")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs. 1/Error")
    plt.legend()
    plt.grid()

    plt.show()



# %%

if __name__ == "__main__":
    # Parameters
    epsilon = 0.1
    sample_sizes = [128, 256, 512, 1024, 2048]

    # Run experiments
    results = run_experiments(sample_sizes, epsilon)

    # Plot results
    plot_results(results)



