
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