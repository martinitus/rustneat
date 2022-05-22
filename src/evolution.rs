use ndarray::{s, Array1, ArrayView};
use population::{Population, Species, Genome};
use compatibility::{DefaultCompatibility, Compatibility};
use rand::prelude::SliceRandom;
use rand::rngs::ThreadRng;
use ndarray_stats::QuantileExt;
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
    fn offspring(&self, population: &Population, fitness: &[f32]) -> Vec<Genome>;

    /// Provide a transformation of the individual fitness values which allows for niching.
    ///
    /// See e.g. https://stackoverflow.com/a/38174559/2160256
    /// Resulting values must be larger or equal to zero.
    fn adjust_fitness(&self, population: &Population, fitness: &[f32]) -> Array1<f32>;

    /// Defines how many offspring the species of the population can produce.
    /// The sum of the returned vector must be equal to the input population size.
    fn budget(&self, population: &Population, fitness: &[f32]) -> Vec<usize>;

    /// Evolve a generation into a new generation.
    ///
    /// Create offspring by mutation and mating. May create new species.
    /// The passed fitness array must be aligned with the populations genome vector.
    /// Returns the next generation of the population.
    fn evolve(&mut self, population: &Population, fitness: &[f32]) -> Population;
}

// const MUTATION_PROBABILITY: f64 = 0.25f64;
// const INTERSPECIE_MATE_PROBABILITY: f64 = 0.001f64;
// const BEST_ORGANISMS_THRESHOLD: f64 = 1f64;
// const MUTATE_CONNECTION_WEIGHT: f64 = 0.90f64;
// const MUTATE_ADD_CONNECTION: f64 = 0.005f64;
// const MUTATE_ADD_NEURON: f64 = 0.004f64;
// const MUTATE_TOGGLE_EXPRESSION: f64 = 0.001f64;
// const MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY: f64 = 0.90f64;
// const MUTATE_TOGGLE_BIAS: f64 = 0.01;


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
    epochs_without_improvements: usize,
    max_epochs_without_improvements: usize,
    species_compatibility_threshold: f64,
    compatibility: DefaultCompatibility,
    rng: ThreadRng,
}

impl Default for DefaultEvolution {
    fn default() -> Self {
        Self {
            epochs_without_improvements: 0,
            max_epochs_without_improvements: 10,
            species_compatibility_threshold: 1., // fixme!
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

    fn offspring(&self, population: &Population, fitness: &[f32]) -> Vec<Genome> {
        // shift fitness such that the lowest value is zero
        let adjusted_fitness = self.adjust_fitness(population, fitness);
        // let foo = adjusted_fitness.as

        // let average_adjusted_fitness = adjusted_fitness.mean().unwrap();
        // adjusted_fitness.clone().quantile_mut(n64(0.5),&Nearest);
        let median_adjusted_fitness = {
            let mut tmp = adjusted_fitness.to_vec();
            tmp.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            tmp[tmp.len() / 2]
        };

        //// gnaaaaa fixme how to determine the number of offspring per individual
        //// based on its adjusted fitness
        //  find the number of offspring per species
        //  sort species according to the individual fitness
        //  mate individuals off the species that are above median fitness
        //  in ascending order (repeatedly) until offspring budget for species is exceeded.
        for (i, (_, _)) in population.iter().enumerate() {
            // check if individual is allowed to reproduce
            if adjusted_fitness[i] > median_adjusted_fitness {
                // let n_offspring = adjusted_fitness[i]
            }
        }
        // let foo: Vec<usize> = (0usize..11).iter().sorted().collet();
        // fo
        // let foo = adjusted_fitness.iter().sorted();
        // let median_adjusted_fitness = adjusted_fitness.clone().quantile_mut(n64(0.5), &Nearest);

        // assert!(&population.species.contains(species));
        //
        // species.genomes.clone()
        //
        // population

        // compute adjusted fitness for each individual
        // let mut adjusted_fitness = (*fitness).clone();
        // for species in population.species() {
        //     // fixme!
        //     // let indices = species.indices();
        //     // for index in &indices {
        //     //     adjusted_fitness[*index] = adjusted_fitness[*index] / indices.len() as f32;
        //     // }
        // }

        // sort individual within their species according to adjusted fitness
        vec![]
    }

    fn adjust_fitness(&self, population: &Population, fitness: &[f32]) -> Array1<f32> {
        let fitness = ArrayView::from(fitness);
        let min = fitness.min_skipnan();
        // divide fitness of each individual by its species size -> adjusted fitness
        Iterator::zip(fitness.iter(), population.iter())
            .map(|(f, (species, _))| (f - min) / species.len() as f32)
            .collect()
    }

    fn budget(&self, population: &Population, fitness: &[f32]) -> Vec<usize> {
        let fitness: Array1<f32> = self.adjust_fitness(population, fitness);
        let mut c = 0;
        let mut species_fitness: Vec<f32> = vec![0.; population.species().len()];
        let mut species_fitness_sum: f32 = 0.;
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
        for genome in self.offspring(population, fitness) {
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
