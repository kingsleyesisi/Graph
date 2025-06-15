from flask import Flask, render_template, request, send_file, redirect, url_for
import base64
import numpy as np
from utils.curve_analysis import detect_curve_type, calculate_slope_stats
from utils.plot_generator import generate_normal_plot, generate_slope_plot
from utils.explanations import generate_slope_explanation
from utils.validators import validate_input_data


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production



# Add built-in functions to Jinja2 template globals
app.jinja_env.globals.update({
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'len': len,
    'sum': sum,
    'int': int,
    'float': float,
    'str': str
})

def ensure_complete_slope_stats(slope_stats, x_values, y_values):
    """
    Ensure slope_stats has all required attributes for the template.
    This prevents jinja2.exceptions.UndefinedError
    """
    if slope_stats is None:
        return None
    
    # Convert to dict if it's not already
    if not isinstance(slope_stats, dict):
        # If it's an object, convert its attributes to dict
        stats_dict = {}
        for attr in dir(slope_stats):
            if not attr.startswith('_'):
                stats_dict[attr] = getattr(slope_stats, attr, None)
        slope_stats = stats_dict
    
    # Ensure all required fields exist with default values
    required_fields = {
        'slope': 0.0,
        'intercept': 0.0,
        'correlation': 0.0,
        'curve_type': 'linear',
        'tangent_point': 0.0,
        'delta_x': 0.0,
        'delta_y': 0.0,
        'x1': 0.0,
        'y1': 0.0,
        'x2': 0.0,
        'y2': 0.0
    }
    
    # Fill in missing fields
    for field, default_value in required_fields.items():
        if field not in slope_stats or slope_stats[field] is None:
            slope_stats[field] = default_value
    
    # Calculate missing values if we have the data
    try:
        if len(x_values) >= 2 and len(y_values) >= 2:
            x_array = np.array(x_values)
            y_array = np.array(y_values)
            
            # If we don't have specific points, use first and last data points
            if slope_stats['x1'] == 0.0 and slope_stats['y1'] == 0.0:
                slope_stats['x1'] = float(x_array[0])
                slope_stats['y1'] = float(y_array[0])
            
            if slope_stats['x2'] == 0.0 and slope_stats['y2'] == 0.0:
                slope_stats['x2'] = float(x_array[-1])
                slope_stats['y2'] = float(y_array[-1])
            
            # Calculate delta values
            slope_stats['delta_x'] = abs(slope_stats['x2'] - slope_stats['x1'])
            slope_stats['delta_y'] = abs(slope_stats['y2'] - slope_stats['y1'])
            
            # Calculate slope if not provided
            if slope_stats['slope'] == 0.0 and slope_stats['delta_x'] != 0:
                slope_stats['slope'] = (slope_stats['y2'] - slope_stats['y1']) / (slope_stats['x2'] - slope_stats['x1'])
            
            # Calculate intercept if not provided
            if slope_stats['intercept'] == 0.0:
                slope_stats['intercept'] = slope_stats['y1'] - slope_stats['slope'] * slope_stats['x1']
            
            # Calculate correlation if not provided
            if slope_stats['correlation'] == 0.0 and len(x_array) > 1:
                correlation_matrix = np.corrcoef(x_array, y_array)
                slope_stats['correlation'] = float(correlation_matrix[0, 1]) if not np.isnan(correlation_matrix[0, 1]) else 0.0
            
            # Set tangent point for curves (middle of x range)
            if slope_stats['tangent_point'] == 0.0:
                slope_stats['tangent_point'] = float(np.mean(x_array))
                
    except Exception as e:
        print(f"Warning: Could not calculate missing slope stats: {e}")
        # Keep default values if calculation fails
        pass
    
    return slope_stats

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        x_str = request.form.get('x_values', '')
        y_str = request.form.get('y_values', '')
        include_slope = request.form.get('include_slope') == 'on'
        
        # Validate inputs using utility function
        x_values, y_values, errors = validate_input_data(x_str, y_str, include_slope)
        
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
                
                # Ensure slope_stats has all required fields
                slope_stats = ensure_complete_slope_stats(slope_stats, x_values, y_values)
                
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
    
    # Validate inputs
    x_values, y_values, errors = validate_input_data(x_str, y_str, include_slope)
    if errors:
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
        print(f"Download error: {e}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)