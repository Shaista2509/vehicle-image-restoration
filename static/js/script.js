const socket = io();

// Emit start signal to begin image processing
socket.emit('start_image_processing');

// Listen for 'image_data' events from the server
socket.on('image_data', (data) => {
    // Update images with the new paths received from the server
    document.getElementById("original_image").src = data.original_image_path;
    document.getElementById("restored_image").src = data.restored_image_path;
    document.getElementById("highlighted_image").src = data.highlighted_image_path;

    // Update metrics
    document.getElementById("psnr").innerText = data.psnr.toFixed(2);
    document.getElementById("ssim").innerText = data.ssim.toFixed(2);

    // Update sensor data text
    document.getElementById("light").innerText = data.sensor_data.light.toFixed(2);
    document.getElementById("temperature").innerText = data.sensor_data.temperature.toFixed(2);
    document.getElementById("humidity").innerText = data.sensor_data.humidity.toFixed(2);

    // Update the sensor data chart
    updateSensorDataChart([
        data.sensor_data.temperature,
        data.sensor_data.light,
        data.sensor_data.humidity
    ]);

});


// Initialize and render the sensor data pie chart
const ctx = document.getElementById('sensorDataChart').getContext('2d');
const sensorDataChart = new Chart(ctx, {
    type: 'pie',
    data: {
        labels: ['Temperature', 'Light', 'Humidity'],
        datasets: [{
            data: [0, 0, 0], // Initial data (will be updated by socket event)
            backgroundColor: ['#ff6384', '#36a2eb', '#ffce56'],
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                display: true,
                position: 'right',
            }
        }
    }
});

// Function to update chart data
function updateSensorDataChart(data) {
    sensorDataChart.data.datasets[0].data = data;
    sensorDataChart.update();
}

// Toggle between Auto and Manual Modes
document.getElementById('autoModeBtn').addEventListener('click', () => {
    document.getElementById('manualAdjustments').style.display = 'none';
    // Enable auto adjustments here
});

document.getElementById('manualModeBtn').addEventListener('click', () => {
    document.getElementById('manualAdjustments').style.display = 'block';
});

// Event listeners for manual adjustments
document.getElementById('brightness').addEventListener('input', (event) => {
    const value = event.target.value;
    // Apply brightness adjustment to the dashboard
    document.body.style.filter = `brightness(${value}%)`;
});

document.getElementById('contrast').addEventListener('input', (event) => {
    const value = event.target.value;
    // Apply contrast adjustment to the dashboard
    document.body.style.filter = `contrast(${value}%)`;
});

document.getElementById('filterStrength').addEventListener('input', (event) => {
    const value = event.target.value;
    // Apply filter strength adjustment to the dashboard
    document.body.style.filter = `grayscale(${value}%)`;
});
