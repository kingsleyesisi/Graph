def validate_input_data(x_str, y_str, include_slope=False):
    """Validate user input and return processed data or errors"""
    errors = []
    
    if not x_str or not y_str:
        errors.append("Both X and Y values are required")
        return None, None, errors
    
    try:
        x_values = [float(x.strip()) for x in x_str.split(',') if x.strip()]
        y_values = [float(y.strip()) for y in y_str.split(',') if y.strip()]
    except ValueError:
        errors.append("Invalid input: Only numbers and commas allowed")
        return None, None, errors
    
    if len(x_values) != len(y_values):
        errors.append("X and Y values must have the same number of points")
        return None, None, errors
    
    if include_slope and len(x_values) < 2:
        errors.append("At least 2 data points are required for slope calculation")
        return None, None, errors
    
    return x_values, y_values, errors