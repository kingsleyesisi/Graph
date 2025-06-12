# app.py
from flask import Flask, render_template, request, send_file, redirect, url_for
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import io
import base64
from scipy import stats
from scipy.optimize import curve_fit
import random
import math

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production

def detect_curve_type(x_values, y_values):
    """
    Detect whether data follows a linear or curved pattern and determine best fit
    Returns: dict with curve type, parameters, and fit quality
    """
    x_array = np.array(x_values)
    y_array = np.array(y_values)
    
    # Test linear fit first
    slope, intercept, r_linear, p_value, std_err = stats.linregress(x_values, y_values)
    r_squared_linear = r_linear ** 2
    
    # Define curve fitting functions
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    def exponential(x, a, b, c):
        return a * np.exp(b * x) + c
    
    def logarithmic(x, a, b):
        return a * np.log(np.abs(x) + 1e-10) + b
    
    def power(x, a, b):
        return a * (np.abs(x) + 1e-10)**b
    
    def inverse(x, a, b):
        return a / (x + 1e-10) + b
    
    # Store all fits
    fits = []
    
    # Linear fit (already calculated)
    fits.append({
        'type': 'linear',
        'function': lambda x, s=slope, i=intercept: s * x + i,
        'equation': f'y = {slope:.4f}x + {intercept:.4f}',
        'r_squared': r_squared_linear,
        'parameters': {'slope': slope, 'intercept': intercept},
        'complexity': 1  # Linear is simplest
    })
    
    # Try curve fits with error handling
    curve_types = [
        ('quadratic', quadratic, lambda a, b, c: f'y = {a:.4f}x² + {b:.4f}x + {c:.4f}'),
        ('exponential', exponential, lambda a, b, c: f'y = {a:.4f}e^({b:.4f}x) + {c:.4f}'),
        ('logarithmic', logarithmic, lambda a, b: f'y = {a:.4f}ln(x) + {b:.4f}'),
        ('power', power, lambda a, b: f'y = {a:.4f}x^{b:.4f}'),
        ('inverse', inverse, lambda a, b: f'y = {a:.4f}/x + {b:.4f}')
    ]
    
    complexities = {'quadratic': 2, 'exponential': 3, 'logarithmic': 2, 'power': 3, 'inverse': 2}
    
    for curve_name, curve_func, eq_formatter in curve_types:
        try:
            # Skip problematic cases
            if curve_name == 'logarithmic' and any(x <= 0 for x in x_values):
                continue
            if curve_name == 'inverse' and any(abs(x) < 1e-10 for x in x_values):
                continue
            if curve_name == 'exponential' and (max(y_values) - min(y_values)) > 1000:
                continue
                
            # Fit curve with reasonable initial guesses
            if curve_name == 'quadratic':
                p0 = [0.1, slope, intercept]
            elif curve_name == 'exponential':
                p0 = [1, 0.1, min(y_values)]
            elif curve_name == 'logarithmic':
                p0 = [1, intercept]
            elif curve_name == 'power':
                p0 = [1, 1]
            elif curve_name == 'inverse':
                p0 = [1, np.mean(y_values)]
            
            popt, _ = curve_fit(curve_func, x_array, y_array, p0=p0, maxfev=2000)

            # Calculate R-squared
            y_pred = curve_func(x_array, *popt)
            ss_res = np.sum((y_array - y_pred) ** 2)
            ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Check for reasonable fit
            bound_func = lambda x, func=curve_func, params=popt: func(x, *params)

            fits.append({
        'type': curve_name,
        'function': bound_func,
        'equation': eq_formatter(*popt),
        'r_squared': r_squared,
        'parameters': popt,
        'complexity': complexities[curve_name]
    })
                
        except (RuntimeError, ValueError, OverflowError, TypeError):
            # Skip fits that fail
            continue
    
    # Select best fit using adjusted R-squared (penalize complexity)
    if len(fits) > 1:
        n = len(x_values)
        for fit in fits:
            k = fit['complexity']  # number of parameters
            if n > k + 1:
                fit['adjusted_r_squared'] = 1 - ((1 - fit['r_squared']) * (n - 1) / (n - k - 1))
            else:
                fit['adjusted_r_squared'] = fit['r_squared']
        
        # Choose best fit (highest adjusted R-squared, but prefer linear if close)
        fits.sort(key=lambda x: x['adjusted_r_squared'], reverse=True)
        best_fit = fits[0]
        
        # Prefer linear if it's reasonably close to the best curve fit
        linear_fit = next(f for f in fits if f['type'] == 'linear')
        if (best_fit['type'] != 'linear' and 
            linear_fit['r_squared'] > 0.85 and 
            best_fit['adjusted_r_squared'] - linear_fit['adjusted_r_squared'] < 0.1):
            best_fit = linear_fit
    else:
        best_fit = fits[0]  # Only linear fit available
    
    return best_fit

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

def calculate_slope_stats(x_values, y_values, curve_info):
    """Calculate slope and statistics based on curve type"""
    if curve_info['type'] == 'linear':
        # Use existing logic for linear
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
        
        return {
            'curve_type': 'linear',
            'slope': calculated_slope,
            'actual_slope': slope_actual,
            'intercept': intercept,
            'correlation': r_value,
            'r_squared': r_value ** 2,
            'std_error': std_err,
            'equation': curve_info['equation'],
            'point1': point1,
            'point2': point2,
            'delta_x': x2 - x1,
            'delta_y': y2 - y1
        }
    else:
        # For curves, we can't use simple slope triangle method
        # Instead, provide tangent slope at a representative point
        x_array = np.array(x_values)
        mid_x = (min(x_values) + max(x_values)) / 2
        
        # Calculate derivative numerically at midpoint
        h = max(0.01 * (max(x_values) - min(x_values)), 0.001)
        try:
            y_plus = curve_info['function'](mid_x + h)
            y_minus = curve_info['function'](mid_x - h)
            tangent_slope = (y_plus - y_minus) / (2 * h)
        except:
            # Fallback to linear approximation if derivative fails
            tangent_slope = (curve_info['function'](mid_x + h) - curve_info['function'](mid_x)) / h
        
        return {
            'curve_type': curve_info['type'],
            'equation': curve_info['equation'],
            'r_squared': curve_info['r_squared'],
            'tangent_slope': tangent_slope,
            'tangent_point': mid_x,
            'tangent_slope': tangent_slope,
            'slope': tangent_slope, 
            'delta_x': None,         # filler so template won’t blow up
            'delta_y': None,
            'is_curve': True
        }

def generate_slope_explanation(slope_stats):
    """Generate educational explanation for slope calculation"""
    if slope_stats['curve_type'] == 'linear':
        # Use existing linear explanation
        x1, y1 = slope_stats['point1']
        x2, y2 = slope_stats['point2']
        
        explanation = f"""
        <div class="slope-explanation bg-blue-50 p-6 rounded-lg border-l-4 border-blue-500 mb-6">
            <h3 class="text-2xl font-bold text-blue-800 mb-4">📊 Linear Slope Analysis (Triangle Method)</h3>
            
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
                    <li><strong>Select two points</strong> on the line at optimal distance</li>
                    <li><strong>Calculate:</strong> Slope = Rise/Run = Δy/Δx = ({slope_stats['delta_y']:.2f})/({slope_stats['delta_x']:.2f}) = {slope_stats['slope']:.4f}</li>
                </ol>
            </div>
        </div>
        """
    else:
        # Curved data explanation
        explanation = f"""
        <div class="slope-explanation bg-purple-50 p-6 rounded-lg border-l-4 border-purple-500 mb-6">
            <h3 class="text-2xl font-bold text-purple-800 mb-4">📈 Curved Data Analysis</h3>
            
            <div class="slope-results bg-white p-4 rounded-lg shadow-sm mb-6">
                <h4 class="text-lg font-semibold text-gray-800 mb-3">🔢 Analysis Results:</h4>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <p><strong>Curve Type:</strong> {slope_stats['curve_type'].title()}</p>
                        <p><strong>Best Fit Equation:</strong> {slope_stats['equation']}</p>
                        <p><strong>R-squared (r²):</strong> {slope_stats['r_squared']:.4f}</p>
                    </div>
                    <div>
                        <p><strong>Tangent Slope at x={slope_stats['tangent_point']:.2f}:</strong> {slope_stats['tangent_slope']:.4f}</p>
                        <p><strong>Data Quality:</strong> {slope_stats['r_squared']*100:.1f}% variance explained</p>
                    </div>
                </div>
            </div>
            
            <div class="slope-method bg-orange-50 p-4 rounded-lg mb-6">
                <h4 class="text-lg font-semibold text-orange-800 mb-3">📐 Understanding Curved Data:</h4>
                <ul class="list-disc list-inside space-y-2 text-gray-700">
                    <li><strong>Non-linear relationship:</strong> The data follows a curved pattern, not a straight line</li>
                    <li><strong>Variable rate of change:</strong> The slope changes at different points along the curve</li>
                    <li><strong>Tangent slope:</strong> At any point, we can find the instantaneous rate of change (tangent)</li>
                    <li><strong>Best fit curve:</strong> The equation shown gives the best mathematical model for your data</li>
                </ul>
            </div>
        </div>
        """
    
    return explanation

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
        
        # Generate normal plot (now with auto curve detection)
        normal_buffer = generate_normal_plot(x_values, y_values)
        normal_plot_data = base64.b64encode(normal_buffer.getvalue()).decode('utf-8')
        
        # Generate slope plot if requested
        slope_plot_data = None
        slope_stats = None
        if include_slope:
            try:
                slope_buffer, slope_stats = generate_slope_plot(x_values, y_values)
                slope_plot_data = base64.b64encode(slope_buffer.getvalue()).decode('utf-8')
            except Exception as e:
                errors.append(f"Error generating slope analysis: {str(e)}")
                return render_template('index.html', errors=errors, x_values=x_str, y_values=y_str)
        
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
    try:
        normal_buffer = generate_normal_plot(x_values, y_values, output_format=format_type)
        
        if include_slope:
            slope_buffer, _ = generate_slope_plot(x_values, y_values, output_format=format_type)
            # For downloads, we'll return the slope graph if slope was requested
            return send_file(
                slope_buffer,
                mimetype='image/png' if format_type == 'png' else 'application/pdf',
                as_attachment=True,
                download_name=f'physics_graph_with_analysis.{format_type}'
            )
        else:
            return send_file(
                normal_buffer,
                mimetype='image/png' if format_type == 'png' else 'application/pdf',
                as_attachment=True,
                download_name=f'physics_graph.{format_type}'
            )
    except Exception as e:
        # Handle errors during download generation
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)