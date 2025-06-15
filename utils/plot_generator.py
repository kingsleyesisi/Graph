import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import io
from .curve_analysis import detect_curve_type, calculate_slope_stats

def generate_normal_plot(x_values, y_values, output_format='png'):
    """Generate a clean physics graph with automatic curve detection"""
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Detect curve type
    curve_info = detect_curve_type(x_values, y_values)
    
    # Plot original data points
    ax.plot(x_values, y_values, 'o', markersize=8, color='blue', label='Data Points')
    
    # Generate smooth curve for fitting
    x_min, x_max = min(x_values), max(x_values)
    x_smooth = np.linspace(x_min, x_max, 200)
    
    try:
        y_smooth = curve_info['function'](x_smooth)
        ax.plot(x_smooth, y_smooth, '-', linewidth=2, color='red', 
                label=f'{curve_info["type"].title()} Fit (R² = {curve_info["r_squared"]:.3f})')
    except:
        # Fallback to connecting points
        ax.plot(x_values, y_values, '-', linewidth=2, color='blue', alpha=0.7, label='Data Connection')
    
    # Add equation as text box
    textstr = f'{curve_info["equation"]}\nR² = {curve_info["r_squared"]:.4f}'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    # Set labels and title
    ax.set_title("Physics Graph with Auto-Detected Fit", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("X Axis", fontsize=16)
    ax.set_ylabel("Y Axis", fontsize=16)
    
    # Add grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
    ax.minorticks_on()
    
    # Auto-scale axes with buffer
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = max(y_values) - min(y_values) if max(y_values) != min(y_values) else 1
    
    ax.set_xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)
    ax.set_ylim(min(y_values) - 0.1*y_range, max(y_values) + 0.1*y_range)
    
    # Add legend
    ax.legend(loc='best', fontsize=10)
    
    # Save to buffer
    buffer = io.BytesIO()
    if output_format == 'pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(buffer) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
    else:
        FigureCanvas(fig).print_png(buffer)
    
    plt.close(fig)
    buffer.seek(0)
    return buffer

def generate_slope_plot(x_values, y_values, output_format='png'):
    """Generate a physics graph with slope/curve analysis"""
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Detect curve type
    curve_info = detect_curve_type(x_values, y_values)
    
    # Calculate slope statistics
    slope_stats = calculate_slope_stats(x_values, y_values, curve_info)
    
    # Plot original data points
    ax.plot(x_values, y_values, 'o', markersize=8, color='blue', label='Data Points')
    
    # Generate smooth curve for fitting
    x_min, x_max = min(x_values), max(x_values)
    x_smooth = np.linspace(x_min, x_max, 200)
    
    try:
        y_smooth = curve_info['function'](x_smooth)
        ax.plot(x_smooth, y_smooth, '--', linewidth=2, color='red', 
                label=f'{curve_info["type"].title()} Fit', alpha=0.8)
    except:
        # If curve plotting fails, skip
        pass
    
    # Add analysis based on curve type
    if slope_stats['curve_type'] == 'linear':
        # Draw triangle for linear data
        x1, y1 = slope_stats['point1']
        x2, y2 = slope_stats['point2']
        
        # Draw the triangle for slope calculation
        ax.plot([x1, x2], [y1, y1], 'g-', linewidth=3, label='Δx (Run)')
        ax.plot([x2, x2], [y1, y2], 'orange', linewidth=3, label='Δy (Rise)')
        ax.plot([x1, x2], [y1, y2], 'purple', linewidth=3, alpha=0.7, label='Slope Line')
        
        # Mark the two selected points
        ax.plot(x1, y1, 'ro', markersize=10, label=f'Point 1 ({x1:.2f}, {y1:.2f})')
        ax.plot(x2, y2, 'ro', markersize=10, label=f'Point 2 ({x2:.2f}, {y2:.2f})')
        
        # Add slope calculation box
        textstr = f'Linear Slope Analysis:\nm = Δy/Δx = {slope_stats["delta_y"]:.2f}/{slope_stats["delta_x"]:.2f} = {slope_stats["slope"]:.4f}'
    else:
        # For curves, show tangent at midpoint
        mid_x = slope_stats['tangent_point']
        try:
            mid_y = curve_info['function'](mid_x)
        except:
            # Fallback to linear approximation
            mid_y = np.mean(y_values)
        
        # Draw tangent line
        tangent_slope = slope_stats['tangent_slope']
        x_range = max(x_values) - min(x_values)
        dx = x_range * 0.2
        
        x_tangent = np.array([mid_x - dx, mid_x + dx])
        y_tangent = mid_y + tangent_slope * (x_tangent - mid_x)
        
        ax.plot(x_tangent, y_tangent, 'g-', linewidth=3, label=f'Tangent at x={mid_x:.2f}')
        ax.plot(mid_x, mid_y, 'ro', markersize=10, label=f'Tangent Point ({mid_x:.2f}, {mid_y:.2f})')
        
        # Add curve analysis box
        textstr = f'Curve Analysis:\nType: {curve_info["type"].title()}\nTangent slope at x={mid_x:.2f}: {tangent_slope:.4f}\nR² = {curve_info["r_squared"]:.4f}'
    
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    # Set labels and title
    title = "Linear Analysis" if slope_stats['curve_type'] == 'linear' else "Curve Analysis"
    ax.set_title(f"Physics Graph with {title}", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("X Axis", fontsize=16)
    ax.set_ylabel("Y Axis", fontsize=16)
    
    # Add grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
    ax.minorticks_on()
    
    # Auto-scale axes with buffer
    x_range = x_max - x_min if x_max != x_min else 1
    y_min, y_max = min(y_values), max(y_values)
    y_range = y_max - y_min if y_max != y_min else 1

    # Apply 15% buffer to axes limits
    x_pad = 0.15 * x_range
    y_pad = 0.15 * y_range


    # add 1 unit extra space on both ends for better visibility
    ax.set_xlim((x_min - x_pad) - 1, (x_max + x_pad) +1)
    ax.set_ylim((y_min - y_pad) - 1, (y_max + y_pad) + 1)
    # Add legend
    ax.legend(loc='best', fontsize=10)
    
    # Save to buffer
    buffer = io.BytesIO()
    if output_format == 'pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(buffer) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
    else:
        FigureCanvas(fig).print_png(buffer)
    
    plt.close(fig)
    buffer.seek(0)
    return buffer, slope_stats