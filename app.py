from flask import Flask, render_template, request, send_file, redirect, url_for
import base64
from utils.curve_analysis import detect_curve_type, calculate_slope_stats
from utils.plot_generator import generate_normal_plot, generate_slope_plot
from utils.explanations import generate_slope_explanation
from utils.validators import validate_input_data

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production

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
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)