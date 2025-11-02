import os
import numpy as np
import matplotlib.pyplot as plt

def plot_house_prices():
    np.random.seed(0)                          # reproducible noise
    years = np.linspace(2000, 2020, 50)        # 50 data points between 2000 and 2020
    prices_trend = np.linspace(10, 200, 50)    # underlying positive trend (millions)
    noise = np.random.normal(scale=5.0, size=50)
    prices = prices_trend + noise              # relatively positive relationship with small noise

    # fit a linear trend line for emphasis
    slope, intercept = np.polyfit(years, prices, 1)
    trend_line = slope * years + intercept
    corr = np.corrcoef(years, prices)[0, 1]

    plt.figure(figsize=(10, 5))
    # draw the data points as a scatter so they are not connected
    plt.scatter(years, prices, marker='o', color='tab:blue', s=35, label='Data')

    # best-fit trend line (for comparison) — show only this line (no illustrative lines)
    # To keep the plotted line exactly the same but show a small intercept value
    # in the label, express the equation around a reference year x0 (use the mean year).
    # show slope and a small intercept-like value (y0 = y at mean year) in the label
    x0 = int(round(years.mean()))
    y0 = slope * x0 + intercept  # value of best-fit at mean year
    plt.plot(years, trend_line, color='tab:orange', linewidth=2,
             label=f'Best fit (slope={slope:.4f})')

    # also plot a new line with a different slope, passing through (x0, y0)
    slope_new = slope * 1.5
    intercept_new = y0 - slope_new * x0  # ensures new line passes through (x0, y0)
    new_line = slope_new * years + intercept_new
    plt.plot(years, new_line, color='tab:green', linestyle='--', linewidth=1.5,
             label=f'New line (slope={slope_new:.4f})')

    # plot a new line with a smaller slope, also passing through (x0, y0)
    slope_small = slope * 0.7
    intercept_small = y0 - slope_small * x0
    new_line_small = slope_small * years + intercept_small
    plt.plot(years, new_line_small, color='tab:purple', linestyle='-.', linewidth=1.5,
             label=f'Smaller slope (slope={slope_small:.4f})')
    plt.xticks(range(2000, 2021, 2))
    plt.ylim(0, 220)
    plt.xlabel('Year')
    plt.ylabel('House Price')
    plt.title(f'House Price Increase (2000–2020) — correlation r={corr:.2f}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # save the full plot (with best-fit and illustrative lines)
    out_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    full_path = os.path.join(out_dir, 'house_prices_with_lines.png')
    plt.savefig(full_path, dpi=150)
    plt.show()

    # create and save a separate figure that shows ONLY the data points (no lines)
    plt.figure(figsize=(10, 5))
    plt.scatter(years, prices, marker='o', color='tab:blue', s=35, label='Data')
    plt.xticks(range(2000, 2021, 2))
    plt.ylim(0, 220)
    plt.xlabel('Year')
    plt.ylabel('House Price')
    plt.title(f'House Price (points only) — correlation r={corr:.2f}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    points_path = os.path.join(out_dir, 'house_prices_points_only.png')
    plt.savefig(points_path, dpi=150)
    plt.show()




    # --- Plot 2D MSE as a function of slope (with optimal intercept for each slope) ---
    x = years
    y = prices
    lines = [
        (slope, intercept, 'Best fit', 'tab:orange', '-'),
        (slope_new, intercept_new, 'New line', 'tab:green', '--'),
        (slope_small, intercept_small, 'Smaller slope', 'tab:purple', '-.'),
    ]
    slopes_range = np.linspace(slope * 0.4, slope * 1.6, 400)
    mses = []
    for m in slopes_range:
        b_opt = y.mean() - m * x.mean()
        preds = m * x + b_opt
        mses.append(np.mean((y - preds) ** 2))
    plt.figure(figsize=(10, 5))
    plt.plot(slopes_range, mses, color='tab:blue', alpha=0.4, label='MSE vs slope (optimal intercept)')
    # Always plot all three lines as points, even if their MSEs overlap
    for m, b, name, color, style in lines:
        mse = np.mean((y - (m * x + b)) ** 2)
        plt.scatter(m, mse, color=color, label=f"{name} (slope={m:.4f}, MSE={mse:.2f})", zorder=5)
        plt.annotate(name, (m, mse), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8, color=color)
    plt.xlabel('Slope')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE for Each Line and Slope')
    plt.legend()
    plt.tight_layout()
    mse2d_path = os.path.join(out_dir, 'mse_lines_2d.png')
    plt.savefig(mse2d_path, dpi=150)
    plt.show()
    print(f'Saved 2D MSE plot:\n - {mse2d_path}')

    # --- Plot 3D MSE as a function of slope and intercept ---
    from mpl_toolkits.mplot3d import Axes3D
    x_mean = x.mean()
    y_mean = y.mean()

    # Use the original intercepts from our first plot to show their true MSE values
    # (This is correct, but let's add a debug print to confirm values)
    print(f"Best fit intercept: {intercept:.4f}, New line intercept: {intercept_new:.4f}, Smaller slope intercept: {intercept_small:.4f}")
    print(f"Best fit slope: {slope:.4f}, New line slope: {slope_new:.4f}, Smaller slope: {slope_small:.4f}")
    print(f"Best fit line: y = {slope:.4f}x + {intercept:.4f}")
    print(f"New line: y = {slope_new:.4f}x + {intercept_new:.4f}")
    print(f"Smaller slope line: y = {slope_small:.4f}x + {intercept_small:.4f}")

    # For the 3D plot, use the same intercept (best-fit) for all lines so only the best-fit line is at the minimum
    lines_3d = [
        (slope, intercept, 'Best fit', 'tab:orange', '-'),
        (slope_new, intercept, 'New line', 'tab:green', '--'),
        (slope_small, intercept, 'Smaller slope', 'tab:purple', '-.'),
    ]

    # Find the range of all three lines' slopes and intercepts
    all_slopes = [slope, slope_new, slope_small]
    all_intercepts = [intercept, intercept_new, intercept_small]
    slope_min, slope_max = min(all_slopes), max(all_slopes)
    intercept_min, intercept_max = min(all_intercepts), max(all_intercepts)

    # Expand ranges to ensure all points are visible
    slope_range = np.linspace(slope_min - 0.2 * (slope_max - slope_min), 
                               slope_max + 0.2 * (slope_max - slope_min), 100)
    intercept_range = np.linspace(intercept_min - 0.2 * (intercept_max - intercept_min), 
                                   intercept_max + 0.2 * (intercept_max - intercept_min), 100)
    S, B = np.meshgrid(slope_range, intercept_range)
    # Use broadcasting for correct 3D MSE bowl
    X = x[None, None, :]  # shape (1, 1, N)
    S_ = S[:, :, None]    # shape (M, M, 1)
    B_ = B[:, :, None]    # shape (M, M, 1)
    Y = y[None, None, :]  # shape (1, 1, N)
    preds = S_ * X + B_
    MSE = np.mean((Y - preds) ** 2, axis=2)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S, B, MSE, cmap='viridis', alpha=0.85, edgecolor='none')
    ax.set_xlabel('Slope')
    ax.set_ylabel('Intercept')
    ax.set_zlabel('Mean Squared Error')
    ax.set_title('MSE as a function of Slope and Intercept')
    fig.colorbar(surf, shrink=0.5, aspect=10, label='MSE')

    # Mark the three lines' (slope, intercept, mse) points
    for m, b, name, color, style in lines_3d:
        preds = m * x + b
        mse_val = np.mean((y - preds) ** 2)
        print(f"{name}: slope={m:.4f}, intercept={b:.4f}, MSE={mse_val:.4f}")
        ax.scatter(m, b, mse_val, color=color, s=100, marker='o', label=f"{name} (slope={m:.4f}, MSE={mse_val:.2f})", zorder=10)
    ax.legend()
    plt.tight_layout()
    mse3d_path = os.path.join(out_dir, 'mse_lines_3d.png')
    plt.savefig(mse3d_path, dpi=150)
    plt.show()
    print(f'Saved 3D MSE plot:\n - {mse3d_path}')



if __name__ == "__main__":
    plot_house_prices()