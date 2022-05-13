use ndarray::{Array1, Array2, s};

use ctrnn::{Ctrnn, CtrnnNeuralNetwork};
use genome::Genome;
use std::cmp;
use std::cmp::Ordering;

/// An organism is a Genome with fitness.
/// Also maintain a fitness measure of the organism.
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct Organism {
    pub genome: Genome,
    pub fitness: f64,
}

impl Ord for Organism {
    fn cmp(&self, other: &Self) -> Ordering {
        other.fitness.partial_cmp(&self.fitness).unwrap()
    }
}

impl Eq for Organism {}

impl PartialEq for Organism {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl PartialOrd for Organism {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Organism {
    /// Create a new organism from a single genome.
    pub fn new(genome: Genome) -> Organism {
        Organism {
            genome: genome,
            fitness: 0f64,
        }
    }
    /// Return a new organism by mutating this Genome and fitness of zero
    pub fn mutate(&self) -> Organism {
        let mut new_genome = self.genome.clone();
        new_genome.mutate();
        Organism::new(new_genome)
    }
    /// Mate this organism with another
    pub fn mate(&self, other: &Organism) -> Organism {
        Organism::new(
            self.genome
                .mate(&other.genome, self.fitness < other.fitness),
        )
    }
    /// Activate this organism in the NN
    pub fn activate(&mut self, sensors: Vec<f64>, outputs: &mut Vec<f64>) {
        let n_neurons = self.genome.len();
        let n_sensors = sensors.len();

        let mut i = sensors.clone();

        if n_neurons < n_sensors {
            i.truncate(n_neurons);
        } else {
            i = [i, vec![0.0; n_neurons - n_sensors]].concat();
        }

        let activations = Ctrnn::default().activate_nn(
            0.1,
            0.01,
            &CtrnnNeuralNetwork {
                y: Array1::zeros((n_neurons, )),
                tau: Array1::from_elem((n_neurons, ), 0.01),
                wji: self.get_weights(),
                theta: self.get_bias(),
                i: i.into(),
            },
        );

        if n_sensors < n_neurons {
            let outputs_activations = activations.slice(s![n_sensors..]).to_vec();

            for n in 0..cmp::min(outputs_activations.len(), outputs.len()) {
                outputs[n] = outputs_activations[n];
            }
        }
    }

    fn get_weights(&self) -> Array2<f64> {
        let neurons_len = self.genome.len();
        let mut matrix = Array2::zeros((neurons_len, neurons_len));
        for gene in self.genome.get_genes() {
            if gene.enabled {
                matrix[[gene.target_id, gene.source_id]] = gene.weight
            }
        }
        matrix
    }

    fn get_bias(&self) -> Array1<f64> {
        let neurons_len = self.genome.len();
        let mut matrix = Array1::zeros((neurons_len, ));
        for gene in self.genome.get_genes() {
            if gene.bias {
                matrix[gene.source_id] += 1f64;
            }
        }
        matrix
    }
}

#[cfg(test)]
use gene::ConnectionGene;


#[cfg(test)]
mod tests {
    use ndarray::array;
    use super::*;
    use genome::Genome;

    #[test]
    fn should_propagate_signal_without_hidden_layers() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 1f64, enabled: true, bias: false });
        let sensors = vec![1.0];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(output[0] > 0.5f64, "{:?} is not bigger than 0.9", output[0]);

        let mut organism = Organism::new(Genome::default());
        organism
            .genome
            .add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: -2f64, enabled: true, bias: false });
        let sensors = vec![1f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(
            output[0] < 0.1f64,
            "{:?} is not smaller than 0.1",
            output[0]
        );
    }

    #[test]
    fn should_propagate_signal_over_hidden_layers() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 0f64, enabled: true, bias: false });
        organism.genome.add_gene(ConnectionGene { source_id: 0, target_id: 2, weight: 5f64, enabled: true, bias: false });
        organism.genome.add_gene(ConnectionGene { source_id: 2, target_id: 1, weight: 5f64, enabled: true, bias: false });
        let sensors = vec![0f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(output[0] > 0.9f64, "{:?} is not bigger than 0.9", output[0]);
    }

    #[test]
    fn should_work_with_cyclic_networks() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 2f64, enabled: true, bias: false });
        organism.genome.add_gene(ConnectionGene { source_id: 1, target_id: 2, weight: 2f64, enabled: true, bias: false });
        organism.genome.add_gene(ConnectionGene { source_id: 2, target_id: 1, weight: 2f64, enabled: true, bias: false });
        let mut output = vec![0f64];
        organism.activate(vec![10f64], &mut output);
        assert!(output[0] > 0.9, "{:#?} is not bigger than 0.9", output[0]);

        let mut organism = Organism::new(Genome::default());
        organism
            .genome
            .add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: -2f64, enabled: true, bias: false });
        organism
            .genome
            .add_gene(ConnectionGene { source_id: 1, target_id: 2, weight: -2f64, enabled: true, bias: false });
        organism
            .genome
            .add_gene(ConnectionGene { source_id: 2, target_id: 1, weight: -2f64, enabled: true, bias: false });
        let mut output = vec![0f64];
        organism.activate(vec![1f64], &mut output);
        assert!(output[0] < 0.1, "{:?} is not smaller than 0.1", output[0]);
    }

    #[test]
    fn activate_organims_sensor_without_enough_neurons_should_ignore_it() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 1f64, enabled: true, bias: false });
        let sensors = vec![0f64, 0f64, 0f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_allow_multiple_output() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 1f64, enabled: true, bias: false });
        let sensors = vec![0f64];
        let mut output = vec![0f64, 0f64];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_be_able_to_get_matrix_representation_of_the_neuron_connections() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 1f64, enabled: true, bias: false });
        organism
            .genome
            .add_gene(ConnectionGene { source_id: 1, target_id: 2, weight: 0.5f64, enabled: true, bias: false });
        organism
            .genome
            .add_gene(ConnectionGene { source_id: 2, target_id: 1, weight: 0.5f64, enabled: true, bias: false });
        organism
            .genome
            .add_gene(ConnectionGene { source_id: 2, target_id: 2, weight: 0.75f64, enabled: true, bias: false });
        organism.genome.add_gene(ConnectionGene { source_id: 1, target_id: 0, weight: 1f64, enabled: true, bias: false });
        assert_eq!(
            organism.get_weights(),
            array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.5], [0.0, 0.5, 0.75]]
        );
    }

    #[test]
    fn should_not_raise_exception_if_less_neurons_than_required() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 1f64, enabled: true, bias: false });
        let sensors = vec![0f64, 0f64, 0f64];
        let mut output = vec![0f64, 0f64, 0f64];
        organism.activate(sensors, &mut output);
    }
}
