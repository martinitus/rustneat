use std::cmp;
use rand::Rng;
use rand::seq::SliceRandom;
use cpython::py_class::CompareOp::Ge;
use population::ConnectionGene;

/// The neat genome is represented as a vector of connection genes.
///
/// For now only contains connection genes.
/// Neuron index 0: Bias neuron
/// Neuron indices 1..n_input+1 : Input neurons
/// Neuron indices n_input+1..n_input+1+n_output : Output neurons
/// Neuron
#[derive(Debug, Clone)]
pub struct Genome {
    /// All connections of the genome including connections from
    /// and to input and output neurons and bias neuron.
    connections: Vec<ConnectionGene>,
    /// The id of the larges neuron of the genome.
    last_neuron_id: usize,
    /// The number of input or sensor neurons.
    n_inputs: usize,
    /// The number of output  neurons.
    n_outputs: usize,
}

const MUTATE_CONNECTION_WEIGHT: f64 = 0.90f64;
const MUTATE_ADD_CONNECTION: f64 = 0.005f64;
const MUTATE_ADD_NEURON: f64 = 0.004f64;
const MUTATE_TOGGLE_EXPRESSION: f64 = 0.001f64;
const MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY: f64 = 0.90f64;
const MUTATE_TOGGLE_BIAS: f64 = 0.01;

impl Genome {
    /// Initialize a genome as a fully connected input/output graph and a disconnected bias neuron.
    pub fn new(inputs: usize, outputs: usize) -> Genome {
        let mut genes: Vec<ConnectionGene> = Vec::default();
        let mut c = 0;
        for i in 1..inputs + 1 {
            for o in inputs + 1..inputs + 1 + outputs {
                genes.push(ConnectionGene::new(i, o, c));
                c += 1;
            }
        }
        Genome { connections: genes, n_inputs: inputs, n_outputs: outputs, last_neuron_id: inputs + outputs }
    }

    /// Mutate the genome.
    /// - may connect two previously disconnected neurons with a random weight
    /// - may add a neuron in an already existing connections splitting the existing connection into
    ///   two. The old connection is disabled, the connection leading into the neuron will be
    ///   initialized with weight 1, the connection from the new neuron will receive the weight of the
    ///   disabled connection.
    /// - may modify the weights of existing neurons.
    pub fn mutate(&mut self, innovation_number: usize) {
        let mut rng = rand::thread_rng();

        // We know the genome is never empty at this point
        if rng.gen_bool(MUTATE_ADD_NEURON) {
            // find connection to be split
            let gene = self.connections.choose_mut(&mut rng).unwrap();
            self.last_neuron_id += 1;
            gene.disable();

            // generate new connections
            let first = ConnectionGene { source_id: gene.source_id, target_id: self.last_neuron_id, weight: 1f64, enabled: true, innovation_id: innovation_number };
            let second = ConnectionGene { source_id: self.last_neuron_id, target_id: gene.target_id, weight: gene.weight, enabled: true, innovation_id: innovation_number + 1 };
            self.connections.push(first);
            self.connections.push(second);
        };

        // try to add a connection between two previously non-connected neurons.
        if rng.gen_bool(MUTATE_ADD_CONNECTION) {
            // fixme: find disconnected pair of neurons.
            let source_id = 0;
            let target_id = 0;

            self.connections.push(ConnectionGene { source_id, target_id, weight: 1., enabled: true, innovation_id: innovation_number });
        };

        if rng.gen_bool(MUTATE_CONNECTION_WEIGHT) {
            for gene in &mut self.connections {
                if rng.gen_bool(MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY) {
                    gene.weight += rng.gen_range(-1.0..1.0);
                } else {
                    gene.weight = rng.gen_range(-1.0..1.0);
                }
            }
        };

        if rng.gen_bool(MUTATE_TOGGLE_EXPRESSION) {
            self.connections.choose_mut(&mut rng).unwrap().toggle_enabled();
        };
    }

    /// Mate two genes
    pub fn mate(&self, other: &Genome, fittest: bool) -> Genome {
        if fittest {
            self.mate_genes(other)
        } else {
            other.mate_genes(self)
        }
    }

    fn mate_genes(&self, other: &Genome) -> Genome {
        let mut genome = Genome::default();
        for gene in &self.connections {
            genome.add_gene({
                //Only mate half of the genes randomly
                if rand::random::<f64>() > 0.5f64 {
                    *gene
                } else {
                    match other.connections.binary_search(gene) {
                        Ok(position) => other.connections[position],
                        Err(_) => *gene,
                    }
                }
            });
        }
        genome
    }

    /// Add a new gene and checks if is allowed. Only can connect next neuron or already connected
    /// neurons.
    // fn add_connection(&mut self, source_id: usize, target_id: usize) {
    //     let gene = ConnectionGene::new(source_id, target_id);
    //     let max_neuron_id = self.last_neuron_id + 1;
    //
    //     if gene.source_id == gene.target_id && gene.source_id > max_neuron_id {
    //         panic!(
    //             "Try to create a gene neuron unconnected, max neuron id {}, {} -> {}",
    //             max_neuron_id,
    //             gene.source_id,
    //             gene.target_id
    //         );
    //     }
    //
    //     if gene.source_id > self.last_neuron_id {
    //         self.last_neuron_id = gene.source_id;
    //     }
    //     if gene.target_id > self.last_neuron_id {
    //         self.last_neuron_id = gene.target_id;
    //     }
    //     match self.connections.binary_search(&gene) {
    //         Ok(pos) => self.connections[pos].enable(),
    //         Err(_) => self.connections.push(gene),
    //     }
    //     self.connections.sort();
    // }

    /// Total weights of all genes
    pub fn total_weights(&self) -> f64 {
        let mut total = 0f64;
        for gene in &self.connections {
            total += gene.weight;
        }
        total
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutation_connection_weight() {
        let mut genome = Genome::default();
        genome.add_gene(ConnectionGene { source_id: 0, target_id: 0, weight: 1f64, enabled: true, bias: false });
        let orig_gene = genome.genes[0];
        genome.mutate_connection_weight();
        // These should not be same size
        assert!((genome.genes[0].weight - orig_gene.weight).abs() > f64::EPSILON);
    }

    #[test]
    fn mutation_add_connection() {
        let mut genome = Genome::default();
        genome.add_connection(1, 2);

        assert_eq!(genome.genes[0].source_id, 1);
        assert_eq!(genome.genes[0].target_id, 2);
    }

    #[test]
    fn mutation_add_neuron() {
        let mut genome = Genome::default();
        genome.mutate_add_connection();
        genome.mutate_add_neuron();
        assert!(!genome.genes[0].enabled);
        assert_eq!(genome.genes[1].source_id, genome.genes[0].source_id);
        assert_eq!(genome.genes[1].target_id, 1);
        assert_eq!(genome.genes[2].source_id, 1);
        assert_eq!(genome.genes[2].target_id, genome.genes[0].target_id);
    }

    #[test]
    #[should_panic(expected = "Try to create a gene neuron unconnected, max neuron id 1, 2 -> 2")]
    fn try_to_inject_a_unconnected_neuron_gene_should_panic() {
        let mut genome1 = Genome::default();
        genome1.add_gene(ConnectionGene { source_id: 2, target_id: 2, weight: 0.5f64, enabled: true, bias: false });
    }

    #[test]
    fn two_genomes_without_differences_should_be_in_same_specie() {
        let mut genome1 = Genome::default();
        genome1.add_gene(ConnectionGene { source_id: 0, target_id: 0, weight: 1f64, enabled: true, bias: false });
        genome1.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 1f64, enabled: true, bias: false });
        let mut genome2 = Genome::default();
        genome2.add_gene(ConnectionGene { source_id: 0, target_id: 0, weight: 0f64, enabled: true, bias: false });
        genome2.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 0f64, enabled: true, bias: false });
        genome2.add_gene(ConnectionGene { source_id: 0, target_id: 2, weight: 0f64, enabled: true, bias: false });
        assert!(genome1.is_same_species(&genome2));
    }

    #[test]
    fn two_genomes_with_enough_difference_should_be_in_different_species() {
        let mut genome1 = Genome::default();
        genome1.add_gene(ConnectionGene { source_id: 0, target_id: 0, weight: 1f64, enabled: true, bias: false });
        genome1.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 1f64, enabled: true, bias: false });
        let mut genome2 = Genome::default();
        genome2.add_gene(ConnectionGene { source_id: 0, target_id: 0, weight: 5f64, enabled: true, bias: false });
        genome2.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 5f64, enabled: true, bias: false });
        genome2.add_gene(ConnectionGene { source_id: 0, target_id: 2, weight: 1f64, enabled: true, bias: false });
        genome2.add_gene(ConnectionGene { source_id: 0, target_id: 3, weight: 1f64, enabled: true, bias: false });
        assert!(!genome1.is_same_species(&genome2));
    }

    #[test]
    fn already_existing_gene_should_be_not_duplicated() {
        let mut genome1 = Genome::default();
        genome1.add_gene(ConnectionGene { source_id: 0, target_id: 0, weight: 1f64, enabled: true, bias: false });
        genome1.add_connection(0, 0);
        assert_eq!(genome1.genes.len(), 1);
        assert!((genome1.get_genes()[0].weight - 1f64).abs() < f64::EPSILON);
    }

    #[test]
    fn adding_an_existing_gene_disabled_should_enable_original() {
        let mut genome1 = Genome::default();
        genome1.add_gene(ConnectionGene { source_id: 0, target_id: 1, weight: 0f64, enabled: true, bias: false });
        genome1.mutate_add_neuron();
        assert!(!genome1.genes[0].enabled);
        assert_eq!(genome1.genes.len(), 3);
        genome1.add_connection(0, 1);
        assert!(genome1.genes[0].enabled);
        assert!((genome1.genes[0].weight - 0f64).abs() < f64::EPSILON);
        assert_eq!(genome1.genes.len(), 3);
    }

    #[test]
    fn genomes_with_same_genes_with_little_differences_on_weight_should_be_in_same_specie() {
        let mut genome1 = Genome::default();
        genome1.add_gene(ConnectionGene { source_id: 0, target_id: 0, weight: 16f64, enabled: true, bias: false });
        let mut genome2 = Genome::default();
        genome2.add_gene(ConnectionGene { source_id: 0, target_id: 0, weight: 16.1f64, enabled: true, bias: false });
        assert!(genome1.is_same_species(&genome2));
    }

    #[test]
    fn genomes_with_same_genes_with_big_differences_on_weight_should_be_in_other_specie() {
        let mut genome1 = Genome::default();
        genome1.add_gene(ConnectionGene { source_id: 0, target_id: 0, weight: 5f64, enabled: true, bias: false });
        let mut genome2 = Genome::default();
        genome2.add_gene(ConnectionGene { source_id: 0, target_id: 0, weight: 15f64, enabled: true, bias: false });
        assert!(!genome1.is_same_species(&genome2));
    }

    #[test]
    fn genomes_initialized_has_correct_neurons() {
        let genome1 = Genome::new(2, 3);
        assert_eq!(genome1.total_genes(), 6);
        assert_eq!(genome1.connections[0].source_id, 0);
        assert_eq!(genome1.connections[0].target_id, 2);
        assert_eq!(genome1.connections[1].source_id, 0);
        assert_eq!(genome1.connections[1].target_id, 3);
        assert_eq!(genome1.connections[2].source_id, 0);
        assert_eq!(genome1.connections[2].target_id, 4);
        assert_eq!(genome1.connections[3].source_id, 1);
        assert_eq!(genome1.connections[3].target_id, 2);
        assert_eq!(genome1.connections[4].source_id, 1);
        assert_eq!(genome1.connections[4].target_id, 3);
        assert_eq!(genome1.connections[5].source_id, 1);
        assert_eq!(genome1.connections[5].target_id, 4);
    }
}

