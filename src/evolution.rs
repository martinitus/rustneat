use ndarray::{s, Array1, ArrayView};
use population::{Population, Species, Genome, Overlay};
use compatibility::{DefaultCompatibility, Compatibility};
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::rngs::ThreadRng;
use ndarray_stats::QuantileExt;
use itertools::Itertools;
use rand::Rng;
// use ndarray_stats::{QuantileExt, Quantile1dExt};
// use ndarray_stats::interpolate::Nearest;
// use noisy_float::prelude::{n64, n32};

pub trait Evolution {
    /// Check if given genome is compatible with any of the given species.
    /// If so, return a mutable reference to the species such that the genome can be added to it.
    /// If no species is compatible a new species will be generated and added to the next generation
    /// of the population.
    fn speciate<'a>(&self, genome: &Genome, species: &'a mut [Species]) -> Option<&'a mut Species>;

    /// Generate offspring for the population.
    /// The fitness corresponds to the fitness of the individuals of the population.
    fn offspring(&mut self, population: &Population, fitness: &[f32]) -> (Overlay, Vec<Genome>);

    /// Provide a transformation of the individual fitness values which allows for niching.
    ///
    /// See e.g. https://stackoverflow.com/a/38174559/2160256
    /// Resulting values must be larger or equal to zero.
    fn adjust_fitness(&self, population: &Population, fitness: &[f32]) -> Array1<f32>;

    // /// Mate and or mutate given parent (with a different genomes of the population or species)
    // /// resulting in a new genome.
    // fn reproduce(&self, parent: &Genome, population: &Population, species: &Species) -> Genome;

    /// Defines how many offspring the species of the population can produce.
    /// The sum of the returned vector must be equal to the input population size.
    /// The elements of the vector correspond to the amount of offspring of the respective
    /// species.
    fn budget(&self, population: &Population, fitness: &[f32]) -> Vec<usize>;

    /// Evolve a generation into a new generation.
    ///
    /// Create offspring by mutation and mating. May create new species.
    /// The passed fitness array must be aligned with the populations genome vector.
    /// Returns the next generation of the population.
    fn evolve(&mut self, population: &Population, fitness: &[f32]) -> Population;

    /// Mutate an individual
    fn mutate(&mut self, genome:  Genome) -> Genome;

    /// Breed a single offspring from two parents
    fn breed(&mut self, genome1: &Genome, genome2: &Genome) -> Genome;
}


/// Implement evolving new individuals and species as close to the original paper as possible.
///
/// Note: The original paper sometimes uses the term compatibility for the parameter δ.
/// However, it is more an "incompatibility" i.e. higher δ means lower compatibility!
///
/// Quote from original paper:
///
/// The distance measure δ (compatibility) allows us to speciate using a compatibility threshold δₜ.
/// An ordered list of species is maintained. In each generation, genomes are sequentially
/// placed into species. Each existing species is represented by a random genome inside
/// the species from the previous generation. A given genome g in the current generation is
/// placed in the first species in which g is compatible with the representative genome of
/// that species. This way, species do not overlap. If g is not compatible with any existing
/// species, a new species is created with g as its representative.
///
/// As the reproduction mechanism for NEAT, we use explicit fitness sharing (Goldberg
/// and Richardson, 1987), where organisms in the same species must share the fitness
/// of their niche. Thus, a species cannot afford to become too big even if many of its
/// organisms perform well. Therefore, any one species is unlikely to take over the entire
/// population, which is crucial for speciated evolution to work. The adjusted fitness fᵢ'
/// for organism i is calculated according to its distance δ from every other organism j in the
/// population:
///
///   fᵢ' = fᵢ / ∑ⱼ H(δ(i,j) - δₜ)
///
/// Where H is the Heaviside Theta function (H(x>0) = 1, H(x<=0) = 0) and δₜ is a predefined
/// compatibility threshold. Thus the sum reduces to the number of organisms in the same species
/// as organism i.
///
/// Every species is assigned a potentially different number of offspring in proportion to the sum
/// of adjusted fitness fᵢ' of its member organisms. Species then reproduce by first eliminating
/// the lowest performing members from the population. The entire population is then replaced by
/// the offspring of the remaining organisms in each species. The net desired effect of speciating
/// the population is to protect topological innovation. The final goal of the system, then, is to
/// perform the search for a solution as efficiently as possible. This goal is achieved through
/// minimizing the dimensionality of the search space.
#[derive(Debug)]
pub struct DefaultEvolution {
    species_compatibility_threshold: f64,
    /// The fitness quantile a genome must reach within a species to be allowed to reproduce.
    /// E.g. for 0.5, the upper 50% of a species are allowed to reproduce if the species has
    /// sufficient budget. If the budget is larger than half the species size, the highest performing
    /// individuals of the species may reproduce multiple times.
    genome_reproduction_threshold: f64,
    /// The fitness quantile a genome must reach to be randomly selected as mating partner.
    genome_reproduction_mating_threshold: f64,
    // mutation_probability: f64,
    /// The probability that offspring is created by a single parent only by mutation.
    mutation_only_probability: f64,
    /// The probability to pick a mating partner from any species (not necessarily the own).
    inter_species_mating_probability: f64,

    /// Probability to add a new node (split existing edge into two)
    mutate_add_node_probability: f64,
    /// Probability to add a new edge
    mutate_add_edge_probability: f64,
    // const BEST_ORGANISMS_THRESHOLD: f64 = 1f64;
// const MUTATE_TOGGLE_EXPRESSION: f64 = 0.001f64;
// const MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY: f64 = 0.90f64;
// const MUTATE_TOGGLE_BIAS: f64 = 0.01;
    compatibility: DefaultCompatibility,
    rng: ThreadRng,
}

impl Default for DefaultEvolution {
    fn default() -> Self {
        Self {
            species_compatibility_threshold: 1., // fixme!
            genome_reproduction_threshold: 0.5,
            genome_reproduction_mating_threshold: 0.5,

            // Original paper: "In each generation, 25% of offspring resulted from mutation without crossover."
            // Not clear how exactly that was done, here, we simply use it as probability
            mutation_only_probability: 0.25,
            inter_species_mating_probability: 0.001,
            mutate_add_node_probability: 0.03,
            mutate_add_edge_probability: 0.3,
            compatibility: DefaultCompatibility::default(),
            rng: Default::default(),
        }
    }
}

impl DefaultEvolution {}


impl Evolution for DefaultEvolution {
    fn speciate<'a>(&self, genome: &Genome, species: &'a mut [Species]) -> Option<&'a mut Species> {
        species.iter_mut().find(|species| {
            let distance = self.compatibility.distance(&genome, &species.representative);
            distance < self.species_compatibility_threshold
        })
    }

    fn offspring(&mut self, population: &Population, fitness: &[f32]) -> (Overlay, Vec<Genome>) {
        // shift fitness such that the lowest value is zero
        // let adjusted_fitness = self.adjust_fitness(population, fitness);	
        // let fitness = ArrayView::from(fitness);

        let _budget = self.budget(population, fitness);
        let overlay = Overlay::new(1, 1); // fixme clone from existing population

        // the offset for each species
        let mut offset: usize = 0;
        let mut offspring: Vec<Genome> = Vec::with_capacity(population.len());

        // fixme: would be nce if self.budget() would return an iterable over species, budget pairs.
        for (budget, species) in _budget.iter().zip(population.species())
        {
            // sort individual within their species according to (adjusted) fitness.
            // It doesn't really matter whether we sort according to their fitness
            // or adjusted fitness, because they are rescaled by the same factor.
            let sorted: Vec<(&Genome, &f32)> = species.genomes
                .iter()
                .zip(&fitness[offset..species.len()])
                .sorted_unstable_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .collect();
            let mut k = 0;
            while &k < budget {
                // get the top genomes from the upper nth quantile of the species and mutate / reproduce them.
                // if the upper nth quantile does not contain b genomes, repeat until b offspring are generated.
                let n = std::cmp::min(
                    f64::ceil(species.len() as f64 * self.genome_reproduction_threshold) as usize,
                    budget.clone(),
                );

                let other_range = &sorted[0..f64::floor(sorted.len() as f64 * self.genome_reproduction_mating_threshold) as usize];
                for i in 0..n {
                    // either take an individual directly for mutation or breed a new individual from two parents.
                    let breed: Genome = {
                        let parent = sorted[i].0;
                        if self.rng.gen_bool(self.mutation_only_probability) {
                            parent.clone()
                        } else {
                            let other = {
                                if self.rng.gen_bool(self.inter_species_mating_probability) || species.len() == 1 {
                                    population.genomes().choose(&mut self.rng).unwrap()
                                } else {
                                    species.iter().choose(&mut self.rng).unwrap()
                                }
                            };
                            self.breed(parent, other)
                        }
                    };
                    offspring.push(self.mutate(breed));
                }

                k = k + n;
            }

            offset += species.len();
        }
        (overlay, offspring)
    }

    // fn reproduce(&self, parent: &Genome, population: &Population, species: &Species) -> Genome {
    //     unimplemented!();
    //     Genome::new(1, 1)
    // }

    fn adjust_fitness(&self, population: &Population, fitness: &[f32]) -> Array1<f32> {
        let fitness = ArrayView::from(fitness);
        let min = fitness.min_skipnan();
        // divide fitness of each individual by its species size -> adjusted fitness
        Iterator::zip(fitness.iter(), population.iter())
            .map(|(f, (species, _))| (f - min) / species.len() as f32)
            .collect()
    }


    /// Fitness sharing equal (or similar?) to original paper:
    /// Calculate the sum over the average adjusted fitness per species.
    /// Then assign budget to each species according to the fraction of the
    /// species' average adjusted fitness to the sum of the adjusted fitness averages.
    fn budget(&self, population: &Population, fitness: &[f32]) -> Vec<usize> {
        let fitness: Array1<f32> = self.adjust_fitness(population, fitness);
        let mut c = 0;
        let mut species_fitness: Vec<f32> = vec![0.; population.species().len()];
        let mut species_fitness_sum: f32 = 0.;

        // calculate average adjusted fitness per species and sum thereof.
        for (i, s) in population.species().iter().enumerate()
        {
            species_fitness[i] = fitness.slice(s![c..s.len()]).mean().unwrap();
            species_fitness_sum += species_fitness[i];
            c += s.len();
        }

        let mut counts: Array1<usize> = species_fitness.iter().map(
            |f| { f32::floor(f * population.len() as f32 / species_fitness_sum) as usize }
        ).collect();

        let delta = counts.sum() as isize - population.len() as isize;

        if delta < 0 {
            for _ in 0..delta.abs() {
                // find smallest species and increment by one
                let smallest = counts.argmin().unwrap();
                counts[smallest] += 1;
            }
        } else if delta > 0 {
            for _ in 0..delta.abs() {
                // find largest species and decrement by one
                let largest = counts.argmax().unwrap();
                counts[largest] -= 1;
            }
        }

        counts.to_vec()
    }

    fn evolve(&mut self, population: &Population, fitness: &[f32]) -> Population {
        assert_eq!(population.len(), fitness.len());

        // copy over existing species but without any individuals
        let mut species: Vec<Species> = population.species().iter().map(
            |s| {
                Species {
                    representative: s.representative.clone(),
                    genomes: Vec::new(),
                    id: s.id,
                }
            }
        ).collect();

        // create offspring from each species
        let (overlay, offspring) = self.offspring(population, fitness);
        for genome in offspring {
            // check if there is a matching species in the old generation
            match self.speciate(&genome, &mut species) {
                // create new species if no match was found
                None => {
                    species.push(Species {
                        representative: genome.clone(),
                        id: population.species.last().unwrap().id,
                        genomes: vec![genome],
                    });
                }
                // otherwise, add genome to existing species
                Some(s) => {
                    s.genomes.push(genome);
                }
            }
        }

        // drop empty species and reassign representatives
        let species: Vec<Species> = species.into_iter()
            .filter(|s| s.genomes.len() > 0)
            .map(|s| {
                Species {
                    representative: s.genomes.choose(&mut self.rng).unwrap().clone(),
                    ..s
                }
            })
            .collect();

        Population {
            n_inputs: population.n_inputs,
            n_outputs: population.n_outputs,
            species,
        }
    }

    /// From the original paper:
    /// > There was an 80% chance of a genome having its connection weights mutated, in which case each weight had a
    /// > 90% chance of being uniformly perturbed and a 10% chance of being assigned a new random value. [...]
    /// > In smaller populations, the probability of adding a new node was 0.03 and the probability of a new link
    /// > mutation was 0.05. In the larger population, the probability of adding a new link was 0.3, because a  larger
    /// > population can tolerate a larger number of prospective species and greater topological diversity
    fn mutate(&mut self, mut genome: Genome) -> Genome {
        if self.rng.gen_bool(0.8) {
            for connection in genome.graph.edge_weights_mut() {
                if self.rng.gen_bool(0.9) {
                    // uniform perturbation
                    connection.gene.weight += self.rng.gen_range(-1f64..1f64)
                } else {
                    // new random value the paper does not seem to specify the distribution...
                    connection.gene.weight = self.rng.gen_range(-1f64..1f64)
                }
            }
        }
        if self.rng.gen_bool(self.mutate_add_node_probability) {
            genome.split()
        }
        if self.rng.gen_bool(self.mutate_add_edge_probability) {}
        genome
    }

    /// From the original paper:
    /// > There was a 75% chance that an inherited gene was disabled if it was disabled in either parent. In each
    /// > generation, 25% of offspring resulted from mutation without crossover. The interspecies mating rate was 0.001.
    fn breed(&mut self, genome1: &Genome, genome2: &Genome) -> Genome {
        todo!();
        genome1.clone()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_evolution() {
        let population = Population::new(10, 2, 2);
        let mut evolution = DefaultEvolution::default();
        let next_gen = evolution.evolve(&population, &[1f32; 10]);
        assert_eq!(next_gen.len(), population.len());
        assert_eq!(next_gen.genomes().next().unwrap().inputs.len(), population.genomes().next().unwrap().inputs.len());
        assert_eq!(next_gen.genomes().next().unwrap().outputs.len(), population.genomes().next().unwrap().outputs.len());
    }

    #[test]
    fn test_total_budget_equals_population_size() {
        let genome = Genome {
            graph: Default::default(),
            bias: Default::default(),
            inputs: vec![],
            outputs: vec![],
        };
        let mut population = Population::new(5, 2, 2);
        population.species = vec![
            Species {
                representative: genome.clone(),
                id: 0,
                genomes: vec![genome.clone(), genome.clone()],
            },
            Species {
                representative: genome.clone(),
                id: 0,
                genomes: vec![genome.clone(), genome.clone(), genome.clone()],
            },
        ];
        let evolution = DefaultEvolution::default();
        let budget = evolution.budget(&population, &[5., 1., 3., 4., 5.]);
        println!("{:?}", budget);
        assert_eq!(budget.iter().fold(0, |acc, &x| acc + x), 5);
    }
}

// /// Mutate the genome.
// /// - may connect two previously disconnected neurons with a random weight
// /// - may add a neuron in an already existing connections splitting the existing connection into
// ///   two. The old connection is disabled, the connection leading into the neuron will be
// ///   initialized with weight 1, the connection from the new neuron will receive the weight of the
// ///   disabled connection.
// /// - may modify the weights of existing neurons.
// pub fn mutate(&mut self, innovation_number: usize) {
//     let mut rng = rand::thread_rng();
//
//     // We know the genome is never empty at this point
//     if rng.gen_bool(MUTATE_ADD_NEURON) {
//         // find connection to be split
//         let gene = self.connections.choose_mut(&mut rng).unwrap();
//         self.last_neuron_id += 1;
//         gene.disable();
//
//         // generate new connections
//         let first = ConnectionGene { source_id: gene.source_id, target_id: self.last_neuron_id, weight: 1f64, enabled: true, innovation_id: innovation_number };
//         let second = ConnectionGene { source_id: self.last_neuron_id, target_id: gene.target_id, weight: gene.weight, enabled: true, innovation_id: innovation_number + 1 };
//         self.connections.push(first);
//         self.connections.push(second);
//     };
//
//     // try to add a connection between two previously non-connected neurons.
//     if rng.gen_bool(MUTATE_ADD_CONNECTION) {
//         // fixme: find disconnected pair of neurons.
//         let source_id = 0;
//         let target_id = 0;
//
//         self.connections.push(ConnectionGene { source_id, target_id, weight: 1., enabled: true, innovation_id: innovation_number });
//     };
//
//     if rng.gen_bool(MUTATE_CONNECTION_WEIGHT) {
//         for gene in &mut self.connections {
//             if rng.gen_bool(MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY) {
//                 gene.weight += rng.gen_range(-1.0..1.0);
//             } else {
//                 gene.weight = rng.gen_range(-1.0..1.0);
//             }
//         }
//     };
//
//     if rng.gen_bool(MUTATE_TOGGLE_EXPRESSION) {
//         self.connections.choose_mut(&mut rng).unwrap().toggle_enabled();
//     };
// }
//
// /// Mate two genes
// pub fn mate(&self, other: &Genome, fittest: bool) -> Genome {
//     if fittest {
//         self.mate_genes(other)
//     } else {
//         other.mate_genes(self)
//     }
// }
//
// fn mate_genes(&self, other: &Genome) -> Genome {
//     let mut genome = Genome::default();
//     for gene in &self.connections {
//         genome.add_gene({
//             //Only mate half of the genes randomly
//             if rand::random::<f64>() > 0.5f64 {
//                 *gene
//             } else {
//                 match other.connections.binary_search(gene) {
//                     Ok(position) => other.connections[position],
//                     Err(_) => *gene,
//                 }
//             }
//         });
//     }
//     genome
// }
// /// Mate and generate offspring, delete old organisms and use the children
// /// as "new" species.
// pub fn generate_offspring(
//     &mut self,
//     num_of_organisms: usize,
//     population_organisms: &[Organism],
// ) {
//     self.age += 1;
//
//     let copy_champion = if num_of_organisms > 5 { 1 } else { 0 };
//
//     let mut organisms_to_mate =
//         (self.organisms.len() as f64 * BEST_ORGANISMS_THRESHOLD) as usize;
//     if organisms_to_mate < 1 {
//         organisms_to_mate = 1;
//     }
//
//     self.organisms.sort();
//     self.organisms.truncate(organisms_to_mate);
//
//     let mut rng = rand::thread_rng();
//     let mut offspring: Vec<Organism> = {
//         let mut selected_organisms = vec![];
//         let range = Range::new(0, self.organisms.len());
//         for _ in 0..num_of_organisms - copy_champion {
//             selected_organisms.push(range.ind_sample(&mut rng));
//         }
//         selected_organisms
//             .iter()
//             .map(|organism_pos| {
//                 self.create_child(&self.organisms[*organism_pos], population_organisms)
//             })
//             .collect::<Vec<Organism>>()
//     };
//
//     if copy_champion == 1 {
//         let champion: Option<Organism> =
//             self.organisms.iter().fold(None, |champion, organism| {
//                 if champion.is_none() || champion.as_ref().unwrap().fitness < organism.fitness {
//                     Some(organism.clone())
//                 } else {
//                     champion
//                 }
//             });
//
//         offspring.push(champion.unwrap());
//     }
//     self.organisms = offspring;
// }
