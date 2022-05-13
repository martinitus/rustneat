use ndarray::prelude::*;
use lazycell::LazyCell;
use std::cmp::max;

/// A connection Gene
#[derive(Debug, Clone, PartialEq)]
pub struct Connection {
    /// The id of the source neuron
    pub source_id: usize,
    /// The id of the target neuron
    pub target_id: usize,
    /// The connection strength
    pub weight: f64,
    /// Whether the connection is enabled or not
    pub enabled: bool,
    /// The innovation number of that gene
    pub innovation_number: usize,
}


/// The neat genome is represented as a vector of connection genes.
///
/// Neuron index 0: Bias neuron. counts as input neuron.
/// Neuron indices 1..n_input+1 : Input neurons
/// Neuron indices n_input+1..n_input+1+n_output : Output neurons
#[derive(Debug, Clone, PartialEq)]
pub struct Genome {
    /// All connections of the genome including connections from
    /// and to input and output neurons and bias neuron.
    /// input_id, output_id, innovation_number.
    /// Ordered by innovation number!
    connections: Vec<Connection>,
}

impl Genome {
    /// Get the connection between given source and target neuron.
    pub fn connection(&self, source: usize, target: usize) -> Option<&'_ Connection> {
        unimplemented!();
        None
    }

    /// Get the connection with given innovation number.
    pub fn innovation(&self, id: usize) -> Option<&'_ Connection> {
        if id > self.connections.last().unwrap().innovation_number {
            return None;
        }

        match self.connections.binary_search_by_key(&id, move |conn| conn.innovation_number) {
            Ok(pos) => Some(&self.connections[pos]),
            Err(_) => None
        }
    }

    /// > Genes that do not match are either disjoint or excess, depending on whether they occur
    /// > within or outside the range of the other parent’s innovation
    /// > numbers. They represent structure that is not present in the other genome.
    /// [Pag. 110, NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
    pub fn excess(&self, other: &Genome) -> impl Iterator<Item=&'_ Connection> {
        let threshold = other.connections.last().unwrap().innovation_number;
        self.connections
            .iter()
            .filter(move |&gene| gene.innovation_number > threshold)
    }
    pub fn disjoint(&self, other: &Genome) -> impl Iterator<Item=&'_ Connection> {
        unimplemented!();
        self.connections.iter()
    }
    pub fn matching(&self, other: &Genome) -> impl Iterator<Item=&'_ Connection> {
        unimplemented!();
        self.connections.iter()
    }

    /// δ = (c1 * E)/N + (c2 * D)/N + c3*W
    pub fn compatibility(&self, other: &Genome) -> f64 {
        // From the original paper:
        // c3 was increased [for DPNV experiment which had population size of 1000 instead of 150]
        // to 3.0 in order to allow for finer distinctions between species based on weight
        // differences (the larger population has room for more species).
        let (c1, c2, c3) = (1., 1., 0.4);

        // Excess count, Disjoint count, and average weight difference of matching genes.
        let n_excess = self.excess(other).count();
        let n_disjoint = self.disjoint(other).count();
        let avg_difference = 0.;

        // N, the number of genes in the larger genome, normalizes for genome size (N
        // can be set to 1 if both genomes are small, i.e., consist of fewer than 20 genes)
        let N = max(self.connections.len(), other.connections.len());

        return (c1 * n_excess as f64 + c2 * n_disjoint as f64) / N as f64 + c3 * avg_difference;
    }
}

#[derive(Debug)]
pub struct Species {
    /// The representative defines whether another
    /// individual belongs to the species or not.
    representative: Genome,
}

impl Species {}

// pub enum Mutation {
//     AddEdge { source_id: int, target_id: int },
//     AddNode(),
//     ChangeWeight(),
//     ToggleExpression(),
// }


const MAX_EPOCHS_WITHOUT_IMPROVEMENTS: usize = 10;

/// A collection of all genomes
///
/// - Has full ownership of all genomes.
/// - Hides storage organization of genomes.
#[derive(Debug)]
pub struct Population {
    /// Outer vector is for the individuals of the population,
    /// Inner vector holds the genes of the respective individual.
    genomes: Vec<Genome>,
    /// The number of input or sensor neurons.
    n_inputs: usize,
    /// The number of output  neurons.
    n_outputs: usize,

    species: Vec<Species>,
    // champion_fitness: f64,
    // epochs_without_improvements: usize,
    // /// champion of the population
    // pub champion: Option<Organism>,
}

impl Population {
    pub fn new(size: usize, inputs: usize, outputs: usize) -> Population {
        Population {
            genomes: vec![Population::initial_genome(inputs, outputs); size],
            n_inputs: inputs,
            n_outputs: outputs,
            //fixme: representative could also be a reference or an index into the genomes array?!
            species: vec![Species { representative: Population::initial_genome(inputs, outputs) }],
        }
    }

    /// Build a fully connected initial (default) genome.
    fn initial_genome(inputs: usize, outputs: usize) -> Genome {
        let mut connections: Vec<Connection> = Vec::default();
        let mut c = 0;
        for i in 0..inputs + 1 {
            for o in inputs + 1..inputs + 1 + outputs {
                connections.push(Connection { source_id: i, target_id: o, weight: 1., enabled: true, innovation_number: c });
                c += 1;
            }
        }
        Genome { connections }
    }

    // pub fn mutate() -> Vec<Mutation> {}

    /// Allow iteration over all species within the population.
    pub fn species(&self) -> &'_ Vec<Species> {
        &self.species
    }

    // /// Allow iteration over the genomes of all individuals of the population.
    pub fn genomes(&self) -> &'_ Vec<Genome> {
        &self.genomes
    }

    /// The total number of individuals in the population.
    pub fn len(&self) -> usize {
        self.genomes.len()
    }

    /// Create offspring by mutation and mating. May create new species.
    /// returns the next generation of the population.
    pub fn evolve(&self, fitness: &Array1<f32>) -> Population {
        let average_fitness = fitness.sum() / fitness.len() as f32;
        let n_organisms = self.len();

        let species = self.species();
        let genomes = self.genomes();

        // if self.epochs_without_improvements > MAX_EPOCHS_WITHOUT_IMPROVEMENTS {
        //     let mut best_species = self.get_best_species();
        //     let num_of_selected = best_species.len();
        //     for specie in &mut best_species {
        //         specie.generate_offspring(
        //             num_of_organisms.checked_div(num_of_selected).unwrap(),
        //             &organisms,
        //         );
        //     }
        //     self.epochs_without_improvements = 0;
        //     return;
        // }

        // let organisms_by_average_fitness =
        //     num_of_organisms.value_as::<f64>().unwrap() / total_average_fitness;

        // for specie in &mut self.species {
        //     let specie_fitness = specie.calculate_average_fitness();
        //     let offspring_size = if total_average_fitness <= 0f64 {
        //         specie.organisms.len()
        //     } else {
        //         (specie_fitness * organisms_by_average_fitness).round() as usize
        //     };
        //     if offspring_size > 0 {
        //         specie.generate_offspring(offspring_size, &organisms);
        //     } else {
        //         specie.remove_organisms();
        //     }
        // }

        // fixme build the next generation
        let next_generation: Vec<Genome> = vec![];

        Population { n_outputs: self.n_outputs, n_inputs: self.n_inputs, genomes: next_generation, species: vec![] }
    }

    // fn get_best_species(&mut self) -> Vec<Species> {
    //     if self.species.len() <= 2 {
    //         return self.species.clone();
    //     }
    //
    //     self.species.sort_by(|specie1, specie2| {
    //         if specie1.calculate_champion_fitness() > specie2.calculate_champion_fitness() {
    //             Ordering::Greater
    //         } else {
    //             Ordering::Less
    //         }
    //     });
    //
    //     self.species[1..2].to_vec().clone()
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_population_should_contain_n_fully_connected_genomes() {
        let p = Population::new(10, 3, 3);
        assert_eq!(p.genomes().len(), 10);
        let sample = &p.genomes()[0];
        for g in p.genomes() {
            assert_eq!(g, sample);
        }

        assert_eq!(sample.connections.len(), 3 * 3 + 3)
        // fixme: add test for connectivity?
    }

    #[test]
    fn new_population_should_contain_one_species() {
        let p = Population::new(10, 3, 3);
        assert_eq!(p.species().len(), 1);
    }
}
