extern crate rand;

use rand::Closed01;
use std::cmp::Ordering;

/// A connection Gene
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "telemetry", derive(Serialize))]
pub struct Gene {
    /// The id of the source neuron
    pub source_id: usize,
    /// The id of the target neuron
    pub target_id: usize,
    /// The connection strength
    pub weight: f64,
    /// Whether the connection is enabled or not
    pub enabled: bool,
    /// Whether the connection is a bias input or not
    pub bias: bool,
}

impl Eq for Gene {}

impl PartialEq for Gene {
    fn eq(&self, other: &Gene) -> bool {
        self.source_id == other.source_id && self.target_id == other.target_id
    }
}

impl Ord for Gene {
    fn cmp(&self, other: &Gene) -> Ordering {
        if self == other {
            Ordering::Equal
        } else if self.source_id == other.source_id {
            if self.target_id > other.target_id {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        } else if self.source_id > other.source_id {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl PartialOrd for Gene {
    fn partial_cmp(&self, other: &Gene) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Gene {
    /// Create a new gene with a specific connection
    pub fn new(in_neuron_id: usize, out_neuron_id: usize) -> Gene {
        Gene {
            source_id: in_neuron_id,
            target_id: out_neuron_id,
            ..Gene::default()
        }
    }

    /// Generate a weight
    pub fn generate_weight() -> f64 {
        rand::random::<Closed01<f64>>().0 * 2f64 - 1f64
    }
    /// Set gene enabled
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    /// Set gene disabled
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Toggle the enable state
    pub fn toggle_enabled(&mut self) { self.enabled = !self.enabled; }
    /// Toggle the bias state
    pub fn toggle_bias(&mut self) { self.bias = !self.bias; }
}

impl Default for Gene {
    fn default() -> Gene {
        Gene {
            source_id: 1,
            target_id: 1,
            weight: Gene::generate_weight(),
            enabled: true,
            bias: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(n_in: usize, n_out: usize) -> Gene {
        Gene {
            source_id: n_in,
            target_id: n_out,
            ..Gene::default()
        }
    }

    #[test]
    fn should_be_able_to_binary_search_for_a_gene() {
        let mut genome = vec![g(0, 1), g(0, 2), g(3, 2), g(2, 3), g(1, 5)];
        genome.sort();
        genome.binary_search(&g(0, 1)).unwrap();
        genome.binary_search(&g(0, 2)).unwrap();
        genome.binary_search(&g(1, 5)).unwrap();
        genome.binary_search(&g(2, 3)).unwrap();
        genome.binary_search(&g(3, 2)).unwrap();
    }
}
