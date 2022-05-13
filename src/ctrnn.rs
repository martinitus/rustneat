use ndarray::prelude::*;

#[cfg(feature = "ctrnn_telemetry")]
use rusty_dashed;

#[cfg(feature = "ctrnn_telemetry")]
use serde_json;

#[allow(missing_docs)]
#[derive(Debug)]
pub struct CTRNN {
    /// Current state of neuron(j)
    pub y: Array1<f64>,
    /// τ - time constant ( t > 0 ). The neuron's speed of response to an external sensory signal.
    /// Membrane resistance time.
    pub tau: Array1<f64>,
    /// Weights of the connection from neuron(j) to neuron(i)
    pub wji: Array2<f64>,
    /// θ - bias of the neuron(j)
    pub theta: Array1<f64>,
}

impl CTRNN {
    /// Activate the NN with given external input and return the output after forward integration.
    pub fn activate_nn(&self, time: f64, step_size: f64, input: &Array1<f64>) -> Array1<f64> {
        let steps = (time / step_size) as usize;
        let mut y = self.y.clone();

        #[cfg(feature = "ctrnn_telemetry")]
            Ctrnn::telemetry(&y);

        for _ in 0..steps {
            let current_weights = (&y + &self.theta).map(&CTRNN::sigmoid);
            y = &y + (((&self.wji.dot(&current_weights)) - &y + input) / &self.tau).map(&|j_value| step_size * j_value);
            #[cfg(feature = "ctrnn_telemetry")]
                Ctrnn::telemetry(&y);
        }
        y
    }

    /// Calculates sigmoid of a number
    pub fn sigmoid(x: &f64) -> f64 {
        1f64 / (1f64 + (-x).exp())
    }

    #[cfg(feature = "ctrnn_telemetry")]
    fn telemetry(y: &Array1<f64>) {
        let y2 = y.clone();
        telemetry!(
            "ctrnn1",
            1.0,
            serde_json::to_string(&y2.into_vec()).unwrap()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::AbsDiffEq;

    #[test]
    fn neural_network_activation_should_return_correct_values() {
        let y = array![0.0, 0.0, 0.0];
        let tau = array![61.694, 10.149, 16.851];
        let wji = array![
            [-2.94737, 2.70665, -0.57046],
            [-3.27553, 3.67193, 1.83218],
            [2.32476, 0.24739, 0.58587],
        ];
        let theta = array![-0.695126, -0.677891, -0.072129];
        let i = array![0.98856, 0.31540, 0.0];

        let nn = CtrnnNeuralNetwork { y, tau, wji, theta, i };

        let ctrnn = Ctrnn::default();

        let result = ctrnn.activate_nn(1.0, 0.1, &nn);
        let expected = array![0.010829986965909134, 0.1324987329841768, 0.06644643156742948];
        assert!(AbsDiffEq::abs_diff_eq(&result, &expected, 1e-8));

        let result = ctrnn.activate_nn(2.0, 0.1, &nn);
        let expected = array![0.02255533337532507, 0.26518982989312406, 0.13038140193371967];
        assert!(AbsDiffEq::abs_diff_eq(&result, &expected, 1e-8));

        let result = ctrnn.activate_nn(10.0, 0.1, &nn);
        let expected = array![0.14934191797049204, 1.3345894864370869, 0.5691613026150651];
        assert!(AbsDiffEq::abs_diff_eq(&result, &expected, 1e-8));

        let result = ctrnn.activate_nn(30.0, 0.1, &nn);
        let expected = array![0.5583616282859531, 3.149231725259237, 1.3050168324825089];
        assert!(AbsDiffEq::abs_diff_eq(&result, &expected, 1e-8));

        let result = ctrnn.activate_nn(100.0, 0.1, &nn);
        let expected = array![1.1121375647080136, 3.43423133661062, 2.0992832630144376];
        // converges
        assert!(AbsDiffEq::abs_diff_eq(&result, &expected, 1e-8));
    }
}
