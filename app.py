# app.py
from flask import Flask, render_template, request, send_file, redirect, url_for
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import io
import base64
from scipy import stats
import random
import math

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production

def find_grid_aligned_points(x_values, y_values, x_range, y_range):
    """Find two points on the best fit line that align with grid subdivisions at optimal distance"""
    # Calculate best fit line parameters
    slope, intercept, _, _, _ = stats.linregress(x_values, y_values)
    
    # Define grid subdivision size
    x_grid_size = (max(x_values) - min(x_values)) / 10  # 10 major divisions
    y_grid_size = (max(y_values) - min(y_values)) / 10  # 10 major divisions
    
    # Generate potential x coordinates that align with grid
    x_min, x_max = min(x_values), max(x_values)
    x_candidates = []
    
    # Create grid-aligned x values
    for i in range(-2, 13):  # Extended range for better point selection
        x_candidate = min(x_values) + i * x_grid_size
        if x_min <= x_candidate <= x_max:
            x_candidates.append(x_candidate)
    
    # Calculate optimal distance range (30-70% of total range)
    total_x_range = x_max - x_min
    min_distance = total_x_range * 0.3  # Minimum 30% of range
    max_distance = total_x_range * 0.7  # Maximum 70% of range
    
    # Find suitable point pairs within the optimal distance range
    suitable_pairs = []
    
    for i in range(len(x_candidates)):
        for j in range(i + 1, len(x_candidates)):
            x1, x2 = x_candidates[i], x_candidates[j]
            x_distance = abs(x2 - x1)
            
            # Check if distance is in optimal range
            if min_distance <= x_distance <= max_distance:
                y1 = slope * x1 + intercept
                y2 = slope * x2 + intercept
                
                # Check if y values are reasonable
                y_min, y_max = min(y_values), max(y_values)
                if y_min <= y1 <= y_max and y_min <= y2 <= y_max:
                    # Round y values to nearest reasonable grid position
                    y1_rounded = round(y1 / y_grid_size) * y_grid_size
                    y2_rounded = round(y2 / y_grid_size) * y_grid_size
                    
                    # Calculate a score based on grid alignment and distance
                    grid_alignment_score = (
                        abs(y1 - y1_rounded) + abs(y2 - y2_rounded)
                    ) / y_grid_size
                    
                    # Prefer points closer to the middle of the optimal range
                    optimal_distance = (min_distance + max_distance) / 2
                    distance_score = abs(x_distance - optimal_distance) / total_x_range
                    
                    # Combined score (lower is better)
                    combined_score = grid_alignment_score + distance_score
                    
                    suitable_pairs.append({
                        'points': ((x1, y1_rounded), (x2, y2_rounded)),
                        'distance': x_distance,
                        'score': combined_score
                    })
    
    # If we have suitable pairs, pick the best one (lowest score)
    if suitable_pairs:
        # Sort by score and pick the best
        suitable_pairs.sort(key=lambda x: x['score'])
        return suitable_pairs[0]['points']
    
    # Fallback: if no suitable pairs found, use points at 40% and 60% of range
    fallback_x1 = x_min + 0.4 * total_x_range
    fallback_x2 = x_min + 0.6 * total_x_range
    fallback_y1 = slope * fallback_x1 + intercept
    fallback_y2 = slope * fallback_x2 + intercept
    
    return ((fallback_x1, fallback_y1), (fallback_x2, fallback_y2))

def calculate_random_slope_stats(x_values, y_values):
    """Calculate slope using two optimally selected grid-aligned points on best fit line"""
    # Get the best fit line first
    slope_actual, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
    
    # Find two suitable points for slope calculation
    x_range = max(x_values) - min(x_values)
    y_range = max(y_values) - min(y_values)
    
    point1, point2 = find_grid_aligned_points(x_values, y_values, x_range, y_range)
    
    # Calculate slope using the two selected points
    x1, y1 = point1
    x2, y2 = point2
    
    # Calculate slope: m = (y2 - y1) / (x2 - x1)
    calculated_slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
    
    # Calculate other statistics
    correlation = r_value
    r_squared = r_value ** 2
    
    return {
        'slope': calculated_slope,
        'actual_slope': slope_actual,  # For comparison
        'intercept': intercept,
        'correlation': correlation,
        'r_squared': r_squared,
        'std_error': std_err,
        'equation': f"y = {calculated_slope:.4f}x + {intercept:.4f}",
        'point1': point1,
        'point2': point2,
        'delta_x': x2 - x1,
        'delta_y': y2 - y1
    }

def generate_slope_explanation(slope_stats):
    """Generate educational explanation for slope calculation with triangle method"""
    x1, y1 = slope_stats['point1']
    x2, y2 = slope_stats['point2']
    
    explanation = f"""
    <div class="slope-explanation bg-blue-50 p-6 rounded-lg border-l-4 border-blue-500 mb-6">
        <h3 class="text-2xl font-bold text-blue-800 mb-4">📊 Slope Analysis (Triangle Method)</h3>
        
        <div class="slope-results bg-white p-4 rounded-lg shadow-sm mb-6">
            <h4 class="text-lg font-semibold text-gray-800 mb-3">🔢 Calculated Results:</h4>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <p><strong>Selected Point 1:</strong> ({x1:.2f}, {y1:.2f})</p>
                    <p><strong>Selected Point 2:</strong> ({x2:.2f}, {y2:.2f})</p>
                    <p><strong>Δx (horizontal change):</strong> {slope_stats['delta_x']:.2f}</p>
                    <p><strong>Δy (vertical change):</strong> {slope_stats['delta_y']:.2f}</p>
                </div>
                <div>
                    <p><strong>Calculated Slope (m):</strong> {slope_stats['slope']:.4f}</p>
                    <p><strong>Linear Equation:</strong> {slope_stats['equation']}</p>
                    <p><strong>Correlation (r):</strong> {slope_stats['correlation']:.4f}</p>
                    <p><strong>R-squared (r²):</strong> {slope_stats['r_squared']:.4f}</p>
                </div>
            </div>
        </div>
        
        <div class="slope-method bg-green-50 p-4 rounded-lg mb-6">
            <h4 class="text-lg font-semibold text-green-800 mb-3">🧮 Triangle Method for Slope Calculation:</h4>
            <ol class="list-decimal list-inside space-y-2 text-gray-700">
                <li><strong>Draw the best fit line</strong> through your data points</li>
                <li><strong>Select two points</strong> on the line at optimal distance (not too close, not too far)</li>
                <li><strong>Ensure coordinates align with grid subdivisions</strong> (avoid fractions when possible)</li>
                <li><strong>Draw a right triangle</strong> using these two points</li>
                <li><strong>Calculate:</strong> Slope = Rise/Run = Δy/Δx = ({slope_stats['delta_y']:.2f})/({slope_stats['delta_x']:.2f}) = {slope_stats['slope']:.4f}</li>
            </ol>
        </div>
        
        <div class="interpretation bg-purple-50 p-4 rounded-lg">
            <h4 class="text-lg font-semibold text-purple-800 mb-3">📈 Physical Interpretation:</h4>
            <p class="text-gray-700 mb-2"><strong>Rate of Change:</strong> The slope represents how much the vertical axis quantity changes per unit change in the horizontal axis quantity.</p>
            <p class="text-gray-700 mb-2"><strong>Slope Value:</strong> {slope_stats['slope']:.4f} means for every 1 unit increase in X, Y changes by {slope_stats['slope']:.4f} units</p>
            <p class="text-gray-700 mb-2"><strong>Relationship Strength:</strong> 
                {'Strong' if abs(slope_stats['correlation']) > 0.8 else 'Moderate' if abs(slope_stats['correlation']) > 0.5 else 'Weak'} 
                {'positive' if slope_stats['correlation'] > 0 else 'negative'} linear relationship (r = {slope_stats['correlation']:.3f})
            </p>
            <p class="text-gray-700"><strong>Data Fit Quality:</strong> {slope_stats['r_squared']*100:.1f}% of the variance is explained by the linear model</p>
        </div>
    </div>
    """
    return explanation

def generate_normal_plot(x_values, y_values, output_format='png'):
    """Generate a clean physics graph without slope analysis"""
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plot original data points
    ax.plot(x_values, y_values, 'o', markersize=8, color='blue', label='Data Points')
    ax.plot(x_values, y_values, '-', linewidth=2, color='blue', alpha=0.7)
    
    # Set labels and title
    ax.set_title("Physics Graph", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("X Axis", fontsize=16)
    ax.set_ylabel("Y Axis", fontsize=16)
    
    # Add grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
    ax.minorticks_on()
    
    # Auto-scale axes with buffer
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    
    ax.set_xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
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
    """Generate a physics graph with slope triangle analysis"""
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plot original data points
    ax.plot(x_values, y_values, 'o', markersize=8, color='blue', label='Data Points')
    ax.plot(x_values, y_values, '-', linewidth=1, color='blue', alpha=0.3)
    
    # Calculate slope statistics using triangle method
    slope_stats = calculate_random_slope_stats(x_values, y_values)
    
    # Generate best fit line
    x_range = np.linspace(min(x_values), max(x_values), 100)
    y_fit = slope_stats['actual_slope'] * x_range + slope_stats['intercept']
    
    # Plot best fit line
    ax.plot(x_range, y_fit, '--', linewidth=2, color='red', 
            label=f'Best Fit Line', alpha=0.8)
    
    # Get the triangle points
    x1, y1 = slope_stats['point1']
    x2, y2 = slope_stats['point2']
    
    # Draw the triangle for slope calculation
    ax.plot([x1, x2], [y1, y1], 'g-', linewidth=3, label='Δx (Run)')
    ax.plot([x2, x2], [y1, y2], 'orange', linewidth=3, label='Δy (Rise)')
    ax.plot([x1, x2], [y1, y2], 'purple', linewidth=3, alpha=0.7, label='Slope Line')
    
    # Mark the two selected points
    ax.plot(x1, y1, 'ro', markersize=10, label=f'Point 1 ({x1:.2f}, {y1:.2f})')
    ax.plot(x2, y2, 'ro', markersize=10, label=f'Point 2 ({x2:.2f}, {y2:.2f})')
    
    # Add annotations for the triangle
    mid_x = (x1 + x2) / 2
    ax.annotate(f'Δx = {slope_stats["delta_x"]:.2f}', 
               xy=(mid_x, y1), xytext=(mid_x, y1 - 0.1*(max(y_values)-min(y_values))),
               ha='center', fontsize=12, fontweight='bold', color='green',
               arrowprops=dict(arrowstyle='->', color='green'))
    
    mid_y = (y1 + y2) / 2
    ax.annotate(f'Δy = {slope_stats["delta_y"]:.2f}', 
               xy=(x2, mid_y), xytext=(x2 + 0.05*(max(x_values)-min(x_values)), mid_y),
               ha='left', va='center', fontsize=12, fontweight='bold', color='orange',
               arrowprops=dict(arrowstyle='->', color='orange'))
    
    # Add slope calculation box
    textstr = f'Slope Calculation:\nm = Δy/Δx = {slope_stats["delta_y"]:.2f}/{slope_stats["delta_x"]:.2f} = {slope_stats["slope"]:.4f}'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    # Set labels and title
    ax.set_title("Physics Graph with Slope Analysis", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("X Axis", fontsize=16)
    ax.set_ylabel("Y Axis", fontsize=16)
    
    # Add grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
    ax.minorticks_on()
    
    # Auto-scale axes with buffer
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    
    ax.set_xlim(x_min - 0.15*x_range, x_max + 0.15*x_range)
    ax.set_ylim(y_min - 0.15*y_range, y_max + 0.15*y_range)
    
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        x_str = request.form.get('x_values', '')
        y_str = request.form.get('y_values', '')
        include_slope = request.form.get('include_slope') == 'on'
        
        # Validate inputs
        errors = []
        if not x_str or not y_str:
            errors.append("Both X and Y values are required")
        
        try:
            x_values = [float(x.strip()) for x in x_str.split(',') if x.strip()]
            y_values = [float(y.strip()) for y in y_str.split(',') if y.strip()]
        except ValueError:
            errors.append("Invalid input: Only numbers and commas allowed")
        
        if not errors and len(x_values) != len(y_values):
            errors.append("X and Y values must have the same number of points")
        
        if not errors and include_slope and len(x_values) < 2:
            errors.append("At least 2 data points are required for slope calculation")
        
        if errors:
            return render_template('index.html', errors=errors, x_values=x_str, y_values=y_str)
        
        # Generate normal plot
        normal_buffer = generate_normal_plot(x_values, y_values)
        normal_plot_data = base64.b64encode(normal_buffer.getvalue()).decode('utf-8')
        
        # Generate slope plot if requested
        slope_plot_data = None
        slope_stats = None
        if include_slope:
            slope_buffer, slope_stats = generate_slope_plot(x_values, y_values)
            slope_plot_data = base64.b64encode(slope_buffer.getvalue()).decode('utf-8')
        
        return render_template('result.html', 
                             normal_plot_data=normal_plot_data,
                             slope_plot_data=slope_plot_data,
                             x_values=x_str, y_values=y_str, 
                             include_slope=include_slope,
                             slope_stats=slope_stats)
    
    return render_template('index.html')

@app.route('/download/<format_type>', methods=['POST'])
def download(format_type):
    if format_type not in ['png', 'pdf']:
        return redirect(url_for('index'))
    
    # Re-process form data
    x_str = request.form['x_values']
    y_str = request.form['y_values']
    include_slope = request.form.get('include_slope') == 'on'
    
    try:
        x_values = [float(x.strip()) for x in x_str.split(',') if x.strip()]
        y_values = [float(y.strip()) for y in y_str.split(',') if y.strip()]
    except ValueError:
        return redirect(url_for('index'))
    
    if len(x_values) != len(y_values):
        return redirect(url_for('index'))
    
    # Generate plots in requested format
    normal_buffer = generate_normal_plot(x_values, y_values, output_format=format_type)
    
    if include_slope:
        slope_buffer, _ = generate_slope_plot(x_values, y_values, output_format=format_type)
        # For downloads, we'll return the slope graph if slope was requested
        return send_file(
            slope_buffer,
            mimetype='image/png' if format_type == 'png' else 'application/pdf',
            as_attachment=True,
            download_name=f'physics_graph_with_slope.{format_type}'
        )
    else:
        return send_file(
            normal_buffer,
            mimetype='image/png' if format_type == 'png' else 'application/pdf',
            as_attachment=True,
            download_name=f'physics_graph.{format_type}'
        )

if __name__ == '__main__':
    app.run(debug=True)