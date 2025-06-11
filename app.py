# app.py
from flask import Flask, render_template, request, send_file, redirect, url_for
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production

def generate_plot(x_values, y_values, output_format='png'):
    """Generate a letter-sized physics graph with automatic scaling"""
    # Create figure with letter size (8.5x11 inches)
    fig = plt.figure(figsize=(8.5, 11), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plot data with points and connecting lines
    ax.plot(x_values, y_values, 'o-', markersize=8, linewidth=2)
    
    # Set labels and title
    ax.set_title("Physics Graph", fontsize=16)
    ax.set_xlabel("X Axis", fontsize=14)
    ax.set_ylabel("Y Axis", fontsize=14)
    
    # Add grid with major and minor lines
    ax.grid(True, which='major', linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3)
    ax.minorticks_on()
    
    # Auto-scale axes with buffer
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    
    ax.set_xlim(x_min - 0.1*x_range, x_max + 0.1*x_range)
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # Save to appropriate format
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        x_str = request.form.get('x_values', '')
        y_str = request.form.get('y_values', '')
        
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
        
        if errors:
            return render_template('index.html', errors=errors, x_values=x_str, y_values=y_str)
        
        # Generate and display plot
        buffer = generate_plot(x_values, y_values)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return render_template('result.html', plot_data=plot_data, 
                             x_values=x_str, y_values=y_str)
    
    return render_template('index.html')

@app.route('/download/<format_type>', methods=['POST'])
def download(format_type):
    if format_type not in ['png', 'pdf']:
        return redirect(url_for('index'))
    
    # Re-process form data
    x_str = request.form['x_values']
    y_str = request.form['y_values']
    
    try:
        x_values = [float(x.strip()) for x in x_str.split(',') if x.strip()]
        y_values = [float(y.strip()) for y in y_str.split(',') if y.strip()]
    except ValueError:
        return redirect(url_for('index'))
    
    if len(x_values) != len(y_values):
        return redirect(url_for('index'))
    
    # Generate plot in requested format
    buffer = generate_plot(x_values, y_values, output_format=format_type)
    return send_file(
        buffer,
        mimetype='image/png' if format_type == 'png' else 'application/pdf',
        as_attachment=True,
        download_name=f'physics_graph.{format_type}'
    )

if __name__ == '__main__':
    app.run(debug=True)