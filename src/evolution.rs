use ndarray::Array1;
use population::Population;

pub trait Evolution {
    /// Evolve a generation into a new generation.
    ///
    /// Create offspring by mutation and mating. May create new species.
    /// The passed fitness array must be aligned with the populations genome vector.
    /// Returns the next generation of the population.
    fn evolve(&self, population: &Population, fitness: &Array1<f32>) -> Population;

    // fixme separate genomes of new population into species?
    // fn speciate(&self);
}

const MUTATION_PROBABILITY: f64 = 0.25f64;
const INTERSPECIE_MATE_PROBABILITY: f64 = 0.001f64;
const BEST_ORGANISMS_THRESHOLD: f64 = 1f64;
const MUTATE_CONNECTION_WEIGHT: f64 = 0.90f64;
const MUTATE_ADD_CONNECTION: f64 = 0.005f64;
const MUTATE_ADD_NEURON: f64 = 0.004f64;
const MUTATE_TOGGLE_EXPRESSION: f64 = 0.001f64;
const MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY: f64 = 0.90f64;
const MUTATE_TOGGLE_BIAS: f64 = 0.01;


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
pub struct DefaultEvolution {
    epochs_without_improvements: usize,
    max_epochs_without_improvements: usize,
}


impl Evolution for DefaultEvolution {
    fn evolve(&self, population: &Population, fitness: &Array1<f32>) -> Population {
        // let species = population.species();
        let genomes = population.genomes();

        // compute adjusted fitness for each individual
        let mut adjusted_fitness = (*fitness).clone();
        for species in population.species() {
            let indices = species.indices();
            for index in &indices {
                adjusted_fitness[*index] = adjusted_fitness[*index] / indices.len() as f32;
            }
        }

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
        //
        // let organisms_by_average_fitness =
        //     num_of_organisms.value_as::<f64>().unwrap() / total_average_fitness;
        //
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
        Population {
            genomes: vec![],
            n_inputs: population.n_inputs,
            n_outputs: population.n_outputs,
            species: vec![],
        }
    }
}

//
// fn generate_offspring(&mut self) {
//     self.speciate();
//
//     let total_average_fitness = self.species.iter_mut().fold(0f64, |total, specie| {
//         total + specie.calculate_average_fitness()
//     });
//
//     let num_of_organisms = self.size();
//     let organisms = self.get_organisms();
//
//     if self.epochs_without_improvements > MAX_EPOCHS_WITHOUT_IMPROVEMENTS {
//         let mut best_species = self.get_best_species();
//         let num_of_selected = best_species.len();
//         for specie in &mut best_species {
//             specie.generate_offspring(
//                 num_of_organisms.checked_div(num_of_selected).unwrap(),
//                 &organisms,
//             );
//         }
//         self.epochs_without_improvements = 0;
//         return;
//     }
//
//     let organisms_by_average_fitness =
//         num_of_organisms.value_as::<f64>().unwrap() / total_average_fitness;
//
//     for specie in &mut self.species {
//         let specie_fitness = specie.calculate_average_fitness();
//         let offspring_size = if total_average_fitness <= 0f64 {
//             specie.organisms.len()
//         } else {
//             (specie_fitness * organisms_by_average_fitness).round() as usize
//         };
//         if offspring_size > 0 {
//             specie.generate_offspring(offspring_size, &organisms);
//         } else {
//             specie.remove_organisms();
//         }
//     }
//     }
//
//     fn get_best_species(&mut self) -> Vec<Specie> {
//         if self.species.len() <= 2 {
//             return self.species.clone();
//         }
//
//         self.species.sort_by(|specie1, specie2| {
//             if specie1.calculate_champion_fitness() > specie2.calculate_champion_fitness() {
//                 Ordering::Greater
//             } else {
//                 Ordering::Less
//             }
//         });
//
//         self.species[1..2].to_vec().clone()
//     }
//
//     fn speciate(&mut self) {
//         let organisms = &self.get_organisms();
//         self.species.retain(|specie| !specie.is_empty());
//
//         let mut next_specie_id = 0i64;
//
//         #[cfg(feature = "telemetry")]
//             let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
//
//         for specie in &mut self.species {
//             #[cfg(feature = "telemetry")]
//             telemetry!(
//                 "species1",
//                 1.0,
//                 format!(
//                     "{{'id':{}, 'fitness':{}, 'organisms':{}, 'timestamp':'{:?}'}}",
//                     specie.id,
//                     specie.calculate_champion_fitness(),
//                     specie.organisms.len(),
//                     now
//                 )
//             );
//
//             specie.choose_new_representative();
//
//             specie.remove_organisms();
//
//             specie.id = next_specie_id;
//             next_specie_id += 1;
//         }
//
//         for organism in organisms {
//             match self
//                 .species
//                 .iter_mut()
//                 .find(|specie| specie.match_genome(organism))
//             {
//                 Some(specie) => {
//                     specie.add(organism.clone());
//                 }
//                 None => {
//                     let mut specie = Specie::new(organism.genome.clone());
//                     specie.id = next_specie_id;
//                     specie.add(organism.clone());
//                     next_specie_id += 1;
//                     self.species.push(specie);
//                 }
//             };
//         }
//         self.species.retain(|specie| !specie.is_empty());
//     }
// }
//
//
//
//
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
