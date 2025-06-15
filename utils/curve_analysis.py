import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

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
