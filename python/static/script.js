document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const videoInput = document.getElementById('video-upload');
    const fileNameDiv = document.getElementById('file-name');
    const alphaRange = document.getElementById('alpha-range');
    const alphaVal = document.getElementById('alpha-val');
    const processBtn = document.getElementById('process-btn');
    const statusSection = document.getElementById('processing-status');
    const resultsSection = document.getElementById('results-section');

    const inputPlayer = document.getElementById('input-player');
    const outputPlayer = document.getElementById('output-player');
    const syncBtn = document.getElementById('sync-btn');

    // Histogram and ROI logic
    const roiCanvas = document.getElementById('roi-canvas');
    const roiCtx = roiCanvas.getContext('2d');
    const histCanvas = document.getElementById('histogram-chart');
    let roi = { x1: 50, y1: 50, x2: 250, y2: 250 }; // Default
    let isSelecting = false;
    let currentInputUrl = '';

    // Initialize Histogram Chart
    const histChart = new Chart(histCanvas, {
        type: 'bar',
        data: {
            labels: Array.from({ length: 256 }, (_, i) => i),
            datasets: [{
                label: 'Pixel Count',
                data: new Array(256).fill(0),
                backgroundColor: 'rgba(88, 166, 255, 0.6)',
                borderColor: 'rgba(88, 166, 255, 1)',
                borderWidth: 1,
                barPercentage: 1.0,
                categoryPercentage: 1.0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
                x: { display: false }
            },
            plugins: { legend: { display: false } },
            animation: false
        }
    });

    // Initialize Frequency Spectrum Chart
    const freqCanvas = document.getElementById('frequency-chart');
    const freqChart = new Chart(freqCanvas, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Magnitude',
                data: [],
                backgroundColor: 'rgba(46, 160, 67, 0.3)',
                borderColor: 'rgba(46, 160, 67, 1)',
                borderWidth: 2,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    title: { display: true, text: 'Magnitude', color: '#8b949e' }
                },
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    title: { display: true, text: 'Frequency (Hz)', color: '#8b949e' }
                }
            },
            plugins: { legend: { display: false } },
            animation: false
        }
    });

    // ROI Drawing
    const drawRoi = () => {
        roiCanvas.width = inputPlayer.clientWidth;
        roiCanvas.height = inputPlayer.clientHeight;
        roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
        roiCtx.strokeStyle = '#58a6ff';
        roiCtx.lineWidth = 2;
        roiCtx.strokeRect(roi.x1, roi.y1, roi.x2 - roi.x1, roi.y2 - roi.y1);
        roiCtx.fillStyle = 'rgba(88, 166, 255, 0.1)';
        roiCtx.fillRect(roi.x1, roi.y1, roi.x2 - roi.x1, roi.y2 - roi.y1);
    };

    roiCanvas.addEventListener('mousedown', (e) => {
        const rect = roiCanvas.getBoundingClientRect();
        roi.x1 = e.clientX - rect.left;
        roi.y1 = e.clientY - rect.top;
        isSelecting = true;
    });

    roiCanvas.addEventListener('mousemove', (e) => {
        if (!isSelecting) return;
        const rect = roiCanvas.getBoundingClientRect();
        roi.x2 = e.clientX - rect.left;
        roi.y2 = e.clientY - rect.top;
        drawRoi();
    });

    const getMappedRoi = () => {
        const videoW = inputPlayer.videoWidth;
        const videoH = inputPlayer.videoHeight;
        const canvasW = roiCanvas.width;
        const canvasH = roiCanvas.height;
        if (videoW === 0 || canvasW === 0) return null;

        const scaleX = videoW / canvasW;
        const scaleY = videoH / canvasH;

        return {
            x1: Math.floor(Math.min(roi.x1, roi.x2) * scaleX),
            y1: Math.floor(Math.min(roi.y1, roi.y2) * scaleY),
            x2: Math.floor(Math.max(roi.x1, roi.x2) * scaleX),
            y2: Math.floor(Math.max(roi.y1, roi.y2) * scaleY)
        };
    };

    roiCanvas.addEventListener('mouseup', async () => {
        isSelecting = false;
        updateFrequencySpectrum();
    });

    window.addEventListener('resize', drawRoi);
    setTimeout(drawRoi, 1000); // Initial draw

    // Update Histogram from Server
    const updateHistogram = async () => {
        if (!currentInputUrl || isSelecting) return;
        const mappedRoi = getMappedRoi();
        if (!mappedRoi) return;

        try {
            const response = await fetch('/histogram', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_path: currentInputUrl,
                    time: inputPlayer.currentTime,
                    roi: mappedRoi
                })
            });
            const data = await response.json();
            if (data.histogram) {
                histChart.data.datasets[0].data = data.histogram;
                histChart.update();
            }
        } catch (e) { console.error(e); }
    };

    const updateFrequencySpectrum = async () => {
        if (!currentInputUrl || isSelecting) return;
        const mappedRoi = getMappedRoi();
        if (!mappedRoi) return;

        try {
            const response = await fetch('/frequency_spectrum', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_path: currentInputUrl,
                    roi: mappedRoi
                })
            });
            const data = await response.json();
            if (data.frequencies) {
                freqChart.data.labels = data.frequencies.map(f => f.toFixed(2));
                freqChart.data.datasets[0].data = data.magnitudes;
                freqChart.update();
            }
        } catch (e) { console.error(e); }
    };

    setInterval(updateHistogram, 500); // Update twice per second

    // Handle form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const file = videoInput.files[0];
        if (!file) return;

        const formData = new FormData(uploadForm);

        // UI state: processing
        processBtn.disabled = true;
        statusSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Update metrics
                document.getElementById('metric-h2d').textContent = `${data.metrics.host_to_device.toFixed(3)} ms`;
                document.getElementById('metric-blur').textContent = `${data.metrics.gaussian_blur.toFixed(3)} ms`;
                document.getElementById('metric-sobel').textContent = `${data.metrics.sobel_x.toFixed(3)} ms`;
                document.getElementById('metric-temp').textContent = `${data.metrics.temporal_filter.toFixed(3)} ms`;
                document.getElementById('metric-amp').textContent = `${data.metrics.amplification.toFixed(3)} ms`;
                document.getElementById('metric-total').textContent = `${data.metrics.total.toFixed(3)} ms`;

                // Update players
                currentInputUrl = data.input_url;
                inputPlayer.src = data.input_url;
                outputPlayer.src = data.output_url;

                // Reset state
                resultsSection.classList.remove('hidden');
                setTimeout(() => {
                    drawRoi();
                    updateFrequencySpectrum();
                }, 500);
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error(error);
            alert('An error occurred during processing.');
        } finally {
            processBtn.disabled = false;
            statusSection.classList.add('hidden');
        }
    });

    // Sync playback
    const syncVideos = () => {
        if (Math.abs(outputPlayer.currentTime - inputPlayer.currentTime) > 0.1) {
            outputPlayer.currentTime = inputPlayer.currentTime;
        }
        if (inputPlayer.paused !== outputPlayer.paused) {
            if (inputPlayer.paused) {
                outputPlayer.pause();
            } else {
                outputPlayer.play().catch(() => { });
            }
        }
    };

    syncBtn.addEventListener('click', syncVideos);

    // Auto-sync when original plays/pauses/seeks
    inputPlayer.addEventListener('play', () => {
        outputPlayer.currentTime = inputPlayer.currentTime;
        outputPlayer.play().catch(() => { });
    });

    inputPlayer.addEventListener('pause', () => {
        outputPlayer.pause();
    });

    inputPlayer.addEventListener('seeked', () => {
        outputPlayer.currentTime = inputPlayer.currentTime;
    });

    inputPlayer.addEventListener('ratechange', () => {
        outputPlayer.playbackRate = inputPlayer.playbackRate;
    });

    // Periodic check to prevent drift
    setInterval(() => {
        if (!inputPlayer.paused && Math.abs(outputPlayer.currentTime - inputPlayer.currentTime) > 0.15) {
            outputPlayer.currentTime = inputPlayer.currentTime;
        }
    }, 1000);
    // Update alpha display
    alphaRange.addEventListener('input', (e) => {
        alphaVal.textContent = e.target.value;
    });

    // Show selected filename
    videoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            fileNameDiv.textContent = file.name;
        }
    });
});
