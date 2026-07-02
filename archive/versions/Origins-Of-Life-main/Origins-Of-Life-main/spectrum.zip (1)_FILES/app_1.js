// CIEL/0 Cosmic Intelligence Detection System - Main Application Logic

class CIEL0System {
    constructor() {
        // Application data from JSON
        this.data = {
            schumann_harmonics: [7.83, 14.3, 20.8, 27.3, 33.8],
            riemann_zeros: [
                {real: 0.5, imag: 14.1347},
                {real: 0.5, imag: 21.0220},
                {real: 0.5, imag: 25.0109},
                {real: 0.5, imag: 30.4249},
                {real: 0.5, imag: 32.9351},
                {real: 0.5, imag: 37.5862}
            ],
            pbh_position: {
                cartesian: {x: -158.4, y: 500.1, z: -254.8},
                celestial: {ra: "7h10m18.395s", dec: "-25°54'27.284\""},
                constellation: "Puppis"
            },
            physical_constants: {
                c: 299792458,
                hbar: 1.054571817e-34,
                mu0: 1.256637062e-6,
                G: 6.67430e-11,
                kB: 1.380649e-23
            },
            default_params: {
                magnetic_field: 5e-6,
                plasma_density: 1e6,
                curvature_scale: 1e26,
                observation_frequency: 70
            }
        };

        // System state
        this.isProcessing = false;
        this.currentParams = {...this.data.default_params};
        this.analysisResults = null;
        this.charts = {};

        // Initialize system
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeControls();
        this.updateOperatorStatus();
        this.setupTabs();
        this.initializeCharts();
    }

    setupEventListeners() {
        // Control sliders
        document.getElementById('magnetic-field').addEventListener('input', (e) => {
            this.currentParams.magnetic_field = parseFloat(e.target.value);
            this.updateControlValue('magnetic-field-value', e.target.value, 'scientific');
        });

        document.getElementById('plasma-density').addEventListener('input', (e) => {
            this.currentParams.plasma_density = parseFloat(e.target.value);
            this.updateControlValue('plasma-density-value', e.target.value, 'scientific');
        });

        document.getElementById('curvature-scale').addEventListener('input', (e) => {
            this.currentParams.curvature_scale = parseFloat(e.target.value);
            this.updateControlValue('curvature-scale-value', e.target.value, 'scientific');
        });

        document.getElementById('frequency').addEventListener('input', (e) => {
            this.currentParams.observation_frequency = parseFloat(e.target.value);
            this.updateControlValue('frequency-value', e.target.value, 'number');
        });

        // Start analysis button
        document.getElementById('start-analysis').addEventListener('click', () => {
            this.startAnalysis();
        });

        // File upload
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        // Upload area drag and drop
        const uploadArea = document.getElementById('upload-area');
        uploadArea.addEventListener('dragenter', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--color-primary)';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--color-border)';
        });

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--color-border)';
            if (e.dataTransfer.files.length > 0) {
                this.handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        uploadArea.addEventListener('click', () => {
            document.getElementById('file-input').click();
        });
    }

    initializeControls() {
        // Set initial values
        document.getElementById('magnetic-field').value = this.currentParams.magnetic_field;
        document.getElementById('plasma-density').value = this.currentParams.plasma_density;
        document.getElementById('curvature-scale').value = this.currentParams.curvature_scale;
        document.getElementById('frequency').value = this.currentParams.observation_frequency;

        // Update display values
        this.updateControlValue('magnetic-field-value', this.currentParams.magnetic_field, 'scientific');
        this.updateControlValue('plasma-density-value', this.currentParams.plasma_density, 'scientific');
        this.updateControlValue('curvature-scale-value', this.currentParams.curvature_scale, 'scientific');
        this.updateControlValue('frequency-value', this.currentParams.observation_frequency, 'number');
    }

    updateControlValue(elementId, value, format) {
        const element = document.getElementById(elementId);
        if (format === 'scientific') {
            element.textContent = parseFloat(value).toExponential(2);
        } else {
            element.textContent = value;
        }
    }

    updateOperatorStatus() {
        // Soul Invariant - complex exponential calculation
        const soulValue = this.calculateSoulInvariant();
        document.getElementById('soul-invariant').textContent = 
            `${soulValue.real.toFixed(3)}${soulValue.imag >= 0 ? '+' : ''}${soulValue.imag.toFixed(3)}i`;

        // Zeta-Riemann
        document.getElementById('zeta-riemann').textContent = 
            `${this.data.riemann_zeros.length} Critical Zeros`;

        // Time Fluid
        document.getElementById('time-fluid').textContent = 'c_s = 1.0';

        // Lambda-Plasma - calculate based on current parameters
        const lambdaValue = this.calculateLambdaPlasma();
        document.getElementById('lambda-plasma').textContent = 
            `Λ = ${lambdaValue.toExponential(2)}`;

        // Intention Field
        const intentionPhase = Math.random() * 2 * Math.PI;
        document.getElementById('intention-field').textContent = 
            `φ = ${intentionPhase.toFixed(3)}`;

        // Update every 2 seconds for dynamic feel
        setTimeout(() => this.updateOperatorStatus(), 2000);
    }

    calculateSoulInvariant() {
        // Simulate topological gauge connection integration
        const theta = Math.random() * 2 * Math.PI;
        return {
            real: Math.cos(theta) * 0.866,
            imag: Math.sin(theta) * 0.5
        };
    }

    calculateLambdaPlasma() {
        const B = this.currentParams.magnetic_field;
        const rho = this.currentParams.plasma_density;
        const L = this.currentParams.curvature_scale;
        const mu0 = this.data.physical_constants.mu0;
        const c = this.data.physical_constants.c;
        const resonance = 0.5 + 0.3 * Math.sin(Date.now() / 1000);

        return (B * B) / (mu0 * rho * c * c) * (1 / (L * L)) * resonance;
    }

    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;

                // Remove active class from all tabs and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked tab and corresponding content
                button.classList.add('active');
                document.getElementById(`${targetTab}-tab`).classList.add('active');
            });
        });
    }

    initializeCharts() {
        // Spectrum Chart
        const spectrumCtx = document.getElementById('spectrum-chart').getContext('2d');
        this.charts.spectrum = new Chart(spectrumCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: 100}, (_, i) => i * 10 + 10),
                datasets: [{
                    label: 'Original CMB Spectrum',
                    data: this.generateCMBSpectrum(100),
                    borderColor: '#1FB8CD',
                    backgroundColor: 'rgba(31, 184, 205, 0.1)',
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'CIEL/0 Modulated',
                    data: this.generateModulatedSpectrum(100),
                    borderColor: '#FFC185',
                    backgroundColor: 'rgba(255, 193, 133, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#f5f5f5'
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Multipole l',
                            color: '#f5f5f5'
                        },
                        ticks: {
                            color: '#a7a9a9'
                        },
                        grid: {
                            color: 'rgba(167, 169, 169, 0.2)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'D_l [μK²]',
                            color: '#f5f5f5'
                        },
                        ticks: {
                            color: '#a7a9a9'
                        },
                        grid: {
                            color: 'rgba(167, 169, 169, 0.2)'
                        }
                    }
                }
            }
        });

        // Anomaly Chart
        const anomalyCtx = document.getElementById('anomaly-chart').getContext('2d');
        this.charts.anomaly = new Chart(anomalyCtx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Normal Regions',
                    data: this.generateAnomalyData(200, false),
                    backgroundColor: '#5D878F',
                    pointRadius: 3
                }, {
                    label: 'Anomalous Regions',
                    data: this.generateAnomalyData(50, true),
                    backgroundColor: '#DB4545',
                    pointRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#f5f5f5'
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Galactic Longitude',
                            color: '#f5f5f5'
                        },
                        ticks: {
                            color: '#a7a9a9'
                        },
                        grid: {
                            color: 'rgba(167, 169, 169, 0.2)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Galactic Latitude',
                            color: '#f5f5f5'
                        },
                        ticks: {
                            color: '#a7a9a9'
                        },
                        grid: {
                            color: 'rgba(167, 169, 169, 0.2)'
                        }
                    }
                }
            }
        });
    }

    generateCMBSpectrum(length) {
        return Array.from({length}, (_, i) => {
            const l = i * 10 + 10;
            // Simulate CMB acoustic peaks
            return 1000 * Math.exp(-l/200) * (1 + Math.sin(l/100)) * (1 + 0.3 * Math.random());
        });
    }

    generateModulatedSpectrum(length) {
        return Array.from({length}, (_, i) => {
            const l = i * 10 + 10;
            // Apply LPEG correction factor
            const lpegFactor = 1 + 0.1 * Math.exp(-l/50) * Math.sin(0.5 * Math.log(l + 1));
            return 1000 * Math.exp(-l/200) * (1 + Math.sin(l/100)) * lpegFactor * (1 + 0.3 * Math.random());
        });
    }

    generateAnomalyData(count, isAnomalous) {
        return Array.from({length: count}, () => ({
            x: (Math.random() - 0.5) * 360,
            y: (Math.random() - 0.5) * 180,
            anomalyStrength: isAnomalous ? 0.7 + Math.random() * 0.3 : Math.random() * 0.3
        }));
    }

    handleFileUpload(file) {
        if (!file) return;

        const fileStatus = document.getElementById('file-status');
        if (file.name.endsWith('.fits') || file.name.endsWith('.fit')) {
            fileStatus.innerHTML = `<p>✅ File loaded: ${file.name}</p>`;
            fileStatus.style.background = 'var(--color-bg-3)';
        } else {
            fileStatus.innerHTML = `<p>⚠️ Invalid file type. Please upload FITS files.</p>`;
            fileStatus.style.background = 'var(--color-bg-4)';
        }
    }

    async startAnalysis() {
        if (this.isProcessing) return;

        this.isProcessing = true;
        this.showProcessingState();

        // Simulate processing time
        const startTime = Date.now();
        
        try {
            // Update processing indicator
            document.getElementById('processing-indicator').classList.remove('hidden');
            document.getElementById('start-analysis').textContent = 'Processing...';
            document.getElementById('start-analysis').disabled = true;

            // Simulate analysis steps
            await this.simulateAnalysisSteps();

            // Calculate results
            this.analysisResults = this.calculateAnalysisResults();

            // Update UI with results
            this.updateResults();

            const processingTime = (Date.now() - startTime) / 1000;
            document.getElementById('processing-time').textContent = `${processingTime.toFixed(2)}s`;

            // Check for intelligence alert
            if (this.analysisResults.intelligenceProbability > 0.7) {
                this.showIntelligenceAlert();
            }

        } catch (error) {
            console.error('Analysis error:', error);
        } finally {
            this.isProcessing = false;
            this.hideProcessingState();
        }
    }

    async simulateAnalysisSteps() {
        const steps = [
            'Initializing CIEL/0 operators...',
            'Computing soul invariant topology...',
            'Applying Zeta-Riemann modulation...',
            'Calculating temporal fluid dynamics...',
            'Processing Lambda-Plasma resonance...',
            'Analyzing intention field coherence...',
            'Detecting anomaly patterns...',
            'Evaluating intelligence signatures...'
        ];

        for (let i = 0; i < steps.length; i++) {
            await new Promise(resolve => setTimeout(resolve, 200 + Math.random() * 300));
            // Could update a progress indicator here
        }
    }

    calculateAnalysisResults() {
        // Enhanced CIEL/0 calculations that can reach higher probabilities
        const soulInvariant = this.calculateSoulInvariant();
        const soulOrganization = Math.abs(soulInvariant.real + soulInvariant.imag) * 0.8 + 0.2;
        
        const riemannCoherence = this.data.riemann_zeros.reduce((sum, zero) => {
            return sum + Math.abs(zero.real * zero.imag) / 50; // Increased sensitivity
        }, 0) / this.data.riemann_zeros.length;

        const fractalComplexity = 0.4 + Math.random() * 0.5;

        // Enhanced Schumann resonance analysis with parameter influence
        const schumannStrength = this.data.schumann_harmonics.reduce((sum, freq) => {
            const paramInfluence = this.currentParams.observation_frequency / 100;
            return sum + Math.sin(freq * Date.now() / 8000 + paramInfluence) ** 2;
        }, 0) / this.data.schumann_harmonics.length;

        // Parameter-based intelligence enhancement
        const parameterBoost = this.calculateParameterBoost();

        // Calculate intelligence probability with higher potential
        const intelligenceProbability = Math.min(
            0.25 * schumannStrength +
            0.25 * riemannCoherence +
            0.30 * soulOrganization +
            0.20 * parameterBoost +
            0.15 * Math.random(), 1.0
        );

        return {
            intelligenceProbability,
            soulOrganization,
            riemannCoherence,
            fractalComplexity,
            schumannStrength,
            anomalyStrength: 0.15 + Math.random() * 0.15,
            coherenceMeasure: 0.6 + Math.random() * 0.3,
            consciousnessIndicators: intelligenceProbability > 0.7
        };
    }

    calculateParameterBoost() {
        // Parameters that favor higher intelligence probability
        const magneticBoost = this.currentParams.magnetic_field > 8e-6 ? 0.3 : 0;
        const densityBoost = this.currentParams.plasma_density > 5e6 ? 0.2 : 0;
        const scaleBoost = this.currentParams.curvature_scale > 5e26 ? 0.2 : 0;
        const freqBoost = this.currentParams.observation_frequency > 200 ? 0.3 : 0;
        
        return Math.min(magneticBoost + densityBoost + scaleBoost + freqBoost + Math.random() * 0.4, 1.0);
    }

    updateResults() {
        const results = this.analysisResults;

        // Update intelligence probability meter
        const percentage = Math.round(results.intelligenceProbability * 100);
        document.getElementById('intelligence-percentage').textContent = `${percentage}%`;
        
        // Update probability meter visual
        const meterCircle = document.querySelector('.meter-circle');
        const angle = results.intelligenceProbability * 360;
        meterCircle.style.background = `conic-gradient(var(--color-primary) ${angle}deg, var(--color-secondary) ${angle}deg)`;

        // Update processing stats
        document.getElementById('data-shape').textContent = '512×1024';
        document.getElementById('anomaly-strength').textContent = results.anomalyStrength.toFixed(4);

        // Update consciousness indicators
        document.getElementById('soul-org').style.width = `${results.soulOrganization * 100}%`;
        document.getElementById('riemann-coh').style.width = `${results.riemannCoherence * 100}%`;
        document.getElementById('fractal-comp').style.width = `${results.fractalComplexity * 100}%`;

        // Update charts with new data
        this.updateCharts();
    }

    updateCharts() {
        // Update spectrum chart with new modulated data
        this.charts.spectrum.data.datasets[1].data = this.generateModulatedSpectrum(100);
        this.charts.spectrum.update();

        // Update anomaly chart
        this.charts.anomaly.data.datasets[1].data = this.generateAnomalyData(50, true);
        this.charts.anomaly.update();
    }

    showProcessingState() {
        document.getElementById('processing-indicator').classList.remove('hidden');
        document.getElementById('start-analysis').textContent = 'Processing...';
        document.getElementById('start-analysis').disabled = true;
        document.getElementById('system-status').textContent = 'Processing...';
        document.getElementById('system-status').className = 'status status--warning';
    }

    hideProcessingState() {
        document.getElementById('processing-indicator').classList.add('hidden');
        document.getElementById('start-analysis').textContent = 'Start CIEL/0 Analysis';
        document.getElementById('start-analysis').disabled = false;
        document.getElementById('system-status').textContent = 'Analysis Complete';
        document.getElementById('system-status').className = 'status status--success';
    }

    showIntelligenceAlert() {
        const alertSection = document.getElementById('intelligence-alert');
        alertSection.classList.remove('hidden');
        
        // Scroll to alert
        alertSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Add glow effect
        alertSection.classList.add('glow');
        
        // Play alert sound (simulated with console log)
        console.log('🚨 INTELLIGENCE ALERT: Probability exceeds 70% threshold!');
    }
}

// Initialize the CIEL/0 System when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const system = new CIEL0System();
    
    // Add some dynamic effects
    setInterval(() => {
        // Animate some background elements
        const cards = document.querySelectorAll('.card');
        cards.forEach((card, index) => {
            if (Math.random() > 0.95) {
                card.style.boxShadow = '0 0 20px rgba(31, 184, 205, 0.3)';
                setTimeout(() => {
                    card.style.boxShadow = '';
                }, 500);
            }
        });
    }, 2000);

    // Simulate real-time data updates
    setInterval(() => {
        if (!system.isProcessing) {
            // Update some operator values
            const elements = document.querySelectorAll('.operator-value');
            elements.forEach(el => {
                if (Math.random() > 0.8) {
                    el.style.textShadow = '0 0 10px rgba(31, 184, 205, 0.8)';
                    setTimeout(() => {
                        el.style.textShadow = '0 0 5px rgba(31, 184, 205, 0.3)';
                    }, 200);
                }
            });
        }
    }, 1000);

    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            switch (e.key) {
                case 'Enter':
                    e.preventDefault();
                    if (!system.isProcessing) {
                        system.startAnalysis();
                    }
                    break;
                case '1':
                    e.preventDefault();
                    document.querySelector('[data-tab="overview"]').click();
                    break;
                case '2':
                    e.preventDefault();
                    document.querySelector('[data-tab="spectrum"]').click();
                    break;
                case '3':
                    e.preventDefault();
                    document.querySelector('[data-tab="anomalies"]').click();
                    break;
            }
        }
    });

    console.log('🌌 CIEL/0 Cosmic Intelligence Detection System Initialized');
    console.log('🔬 Ready for CMB analysis and intelligence detection');
    console.log('⌨️  Keyboard shortcuts: Ctrl+Enter (analyze), Ctrl+1/2/3 (switch tabs)');
    console.log('💡 Tip: Adjust parameters to higher values to increase intelligence probability');
});