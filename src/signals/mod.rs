use std::{collections::HashMap, hash::Hash};

use linfa::Dataset;
use ndarray::{Array1, Array2, Ix1};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

const WINDOW_SIZE: usize = 16;

pub struct Discharge {
    pub class: DisruptionClass,
    pub signals: Vec<Signal>,
}

#[derive(Deserialize, Clone)]
pub struct Signal {
    pub label: String,
    pub _times: Vec<f64>,
    pub values: Vec<f64>,
    pub class: DisruptionClass,
    pub signal_type: SignalType,
    min: f64,
    max: f64,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum DisruptionClass {
    Normal = 0,
    Anomaly = 1,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub enum SignalType {
    CorrientePlasma,
    ModeLock,
    Inductancia,
    Densidad,
    DerivadaEnergiaDiamagnetica,
    PotenciaRadiada,
    PotenciaDeEntrada,
}

impl SignalType {
    pub fn count() -> usize {
        7
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NormalizationParams {
    pub min: HashMap<SignalType, f64>,
    pub max: HashMap<SignalType, f64>,
}

#[derive(Debug)]
pub enum FeatureType {
    Mean,
    FftStd,
}

impl FeatureType {
    pub fn count() -> usize {
        2
    }
}

struct SignalFeatures {
    type_: FeatureType,
    values: Vec<f64>,
}

impl std::fmt::Debug for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Signal {{ label: {}, min: {}, max: {} }}. Values length: {}",
            self.label,
            self.min,
            self.max,
            self.values.len()
        )
    }
}

impl std::fmt::Debug for SignalFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SignalFeatures {{ type: {:?}, values: {:?} }}",
            self.type_, self.values
        )
    }
}

impl Signal {
    pub fn new(
        label: String,
        values: Vec<f64>,
        class: DisruptionClass,
        signal_type: SignalType,
    ) -> Signal {
        let min = values.iter().fold(f64::MAX, |a, b| a.min(*b));
        let max = values.iter().fold(f64::MIN, |a, b| a.max(*b));
        Signal {
            label,
            _times: Vec::new(),
            values,
            class,
            signal_type,
            min,
            max,
        }
    }

    pub fn _normalize(&mut self) {
        self.values
            .iter_mut()
            .for_each(|v| *v = (*v - self.min) / (self.max - self.min));
        self.min = self.values.iter().fold(f64::MAX, |a, b| a.min(*b));
        self.max = self.values.iter().fold(f64::MIN, |a, b| a.max(*b));
    }

    /// Recibe un vector de señales y las normaliza, según la fórmula:
    /// \[ x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} \]
    /// donde \( x_{min} \) y \( x_{max} \) son los valores mínimo y máximo de todas las señales del mismo tipo.
    /// Devuelve un vector con las señales normalizadas.
    pub fn normalize_vec(
        signals: Vec<Signal>,
        params: Option<&NormalizationParams>,
    ) -> (Vec<Signal>, NormalizationParams) {
        let mut signals_norm = Vec::new();

        let mut min_by_type: HashMap<SignalType, f64> = HashMap::new();
        let mut max_by_type: HashMap<SignalType, f64> = HashMap::new();

        if let Some(p) = params {
            min_by_type = p.min.clone();
            max_by_type = p.max.clone();
        } else {
            for signal in &signals {
                let min_entry = min_by_type
                    .entry(signal.signal_type.clone())
                    .or_insert(f64::MAX);
                *min_entry = min_entry.min(signal.min);

                let max_entry = max_by_type
                    .entry(signal.signal_type.clone())
                    .or_insert(f64::MIN);
                *max_entry = max_entry.max(signal.max);
            }
        }

        for signal in signals {
            let global_min = min_by_type.get(&signal.signal_type).unwrap();
            let global_max = max_by_type.get(&signal.signal_type).unwrap();

            let mut values_norm = Vec::new();
            for value in signal.values {
                let value_norm = (value - global_min) / (global_max - global_min);
                values_norm.push(value_norm);
            }

            let local_min = values_norm.iter().fold(f64::MAX, |a, b| a.min(*b));
            let local_max = values_norm.iter().fold(f64::MIN, |a, b| a.max(*b));

            signals_norm.push(Signal {
                label: signal.label,
                _times: signal._times,
                values: values_norm,
                class: signal.class,
                signal_type: signal.signal_type,
                min: local_min,
                max: local_max,
            });
        }

        (
            signals_norm,
            NormalizationParams {
                min: min_by_type,
                max: max_by_type,
            },
        )
    }

    /// Calcula las características de la señal en ventanas de tamaño `window_size`.
    /// Devuelve dos vectores con los valores medios y desviaciones estándar de la FFT.
    /// **Se asume que la señal ya está normalizada.**
    fn get_features(&self, window_size: usize) -> (SignalFeatures, SignalFeatures) {
        let num_windows = self.values.len() / window_size;

        // Vectores para almacenar resultados
        let mut mean_values = Vec::with_capacity(num_windows);
        let mut fft_std_values = Vec::with_capacity(num_windows);

        // Configurar FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(window_size);

        for k in 0..num_windows {
            let start_idx = k * window_size;
            let end_idx = (k + 1) * window_size;
            let window = &self.values[start_idx..end_idx];

            // 1. Calcular valor medio de la ventana
            let mean_value = window.iter().sum::<f64>() / window_size as f64;
            mean_values.push(mean_value);

            // 2. Calcular FFT
            // Convertir a números complejos para la FFT
            let mut fft_input: Vec<Complex<f64>> =
                window.iter().map(|&x| Complex::new(x, 0.0)).collect();

            fft.process(&mut fft_input);

            // Obtener magnitudes de frecuencias positivas (sin DC)
            // Solo necesitamos hasta window_size/2 por simetría
            let magnitudes: Vec<f64> = fft_input[1..=window_size / 2]
                .iter()
                .map(|c| c.norm())
                .collect();

            // Calcular desviación estándar de las magnitudes
            let fft_mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
            let fft_variance = magnitudes
                .iter()
                .map(|&m| (m - fft_mean) * (m - fft_mean))
                .sum::<f64>()
                / magnitudes.len() as f64;

            let fft_std = fft_variance.sqrt();
            fft_std_values.push(fft_std);
        }

        (
            SignalFeatures {
                type_: FeatureType::Mean,
                values: mean_values,
            },
            SignalFeatures {
                type_: FeatureType::FftStd,
                values: fft_std_values,
            },
        )
    }

    pub fn _min(&self) -> f64 {
        self.min
    }

    pub fn _max(&self) -> f64 {
        self.max
    }
}

impl Discharge {
    pub fn new(class: DisruptionClass, signals: Vec<Signal>) -> Discharge {
        Discharge { class, signals }
    }
}

pub fn get_dataset(
    discharges: Vec<Discharge>,
    params: &NormalizationParams,
) -> Dataset<f64, bool, Ix1> {
    // Process each discharge separately and combine the results
    let mut all_records = Vec::new();
    let mut all_targets = Vec::new();

    for discharge in &discharges {
        // Normalize signals for this discharge
        let discharge_signals = discharge.signals.clone();
        let normalized_signals = Signal::normalize_vec(discharge_signals, Some(params)).0;

        // Process signals by type for this discharge
        let mut features_by_type: HashMap<SignalType, (Vec<f64>, Vec<f64>)> = HashMap::new();
        let mut windows_count = 0;

        // Extract features from signals of this discharge
        for signal in &normalized_signals {
            let (mean_features, fft_features) = signal.get_features(WINDOW_SIZE);
            let is_mean_features_empty = mean_features.values.is_empty();
            let mean_features_len = mean_features.values.len();

            // Store feature values by signal type
            features_by_type.insert(
                signal.signal_type.clone(),
                (mean_features.values, fft_features.values),
            );

            // All signals in the same discharge should have the same window count
            if windows_count == 0 && !is_mean_features_empty {
                windows_count = mean_features_len;
            }
        }

        // Create records for this discharge
        for window_idx in 0..windows_count {
            let mut window_features =
                Vec::with_capacity(SignalType::count() * FeatureType::count());
            let is_anomaly = discharge.class == DisruptionClass::Anomaly;

            // Collect features for all signal types in order
            for signal_type in &[
                SignalType::CorrientePlasma,
                SignalType::ModeLock,
                SignalType::Inductancia,
                SignalType::Densidad,
                SignalType::DerivadaEnergiaDiamagnetica,
                SignalType::PotenciaRadiada,
                SignalType::PotenciaDeEntrada,
            ] {
                // Add mean and fft_std values
                if let Some((mean_values, _)) = features_by_type.get(signal_type) {
                    if window_idx < mean_values.len() {
                        window_features.push(mean_values[window_idx]);
                        // We'll add FFT values later to keep them grouped
                    } else {
                        println!("Window index out of bounds for mean values");
                        window_features.push(0.0);
                    }
                } else {
                    println!("Signal type not found in features_by_type");
                    window_features.push(0.0);
                }
            }

            // Now add all the FFT values in the same order
            for signal_type in &[
                SignalType::CorrientePlasma,
                SignalType::ModeLock,
                SignalType::Inductancia,
                SignalType::Densidad,
                SignalType::DerivadaEnergiaDiamagnetica,
                SignalType::PotenciaRadiada,
                SignalType::PotenciaDeEntrada,
            ] {
                if let Some((_, fft_values)) = features_by_type.get(signal_type) {
                    if window_idx < fft_values.len() {
                        window_features.push(fft_values[window_idx]);
                    } else {
                        window_features.push(0.0);
                    }
                } else {
                    window_features.push(0.0);
                }
            }

            // Add this window's data to the combined results
            all_records.push(window_features);
            all_targets.push(is_anomaly);
        }
    }

    // Convert to ndarray structures
    let records = Array2::from_shape_vec(
        (
            all_records.len(),
            SignalType::count() * FeatureType::count(),
        ),
        all_records.into_iter().flatten().collect(),
    )
    .unwrap();

    let targets = Array1::from_vec(all_targets);

    log::info!(
        "Dataset created with {} records and {} targets",
        records.shape()[0],
        targets.len()
    );

    let num_of_true = targets.iter().filter(|&&x| x == true).count();
    log::info!("Number of true targets: {}", num_of_true);

    Dataset::new(records, targets)
}
