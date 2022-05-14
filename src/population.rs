use generator::Gn;
use ndarray::prelude::*;
use lazycell::LazyCell;
use std::cmp::max;
use petgraph::{Directed, Graph};
use petgraph::graph::{EdgeReference, NodeIndex, Node, EdgeIndex};

/// A connection Gene
#[derive(Debug, Clone, PartialEq)]
pub struct Connection {
    /// The connection strength
    pub weight: f64,
    /// Whether the connection is enabled or not
    pub enabled: bool,
    /// The innovation number of that gene
    pub innovation_id: usize,
}


/// The neat genome is represented as a vector of connection genes.
///
/// Neuron index 0: Bias neuron. counts as input neuron.
/// Neuron indices 1..n_input+1 : Input neurons
/// Neuron indices n_input+1..n_input+1+n_output : Output neurons
#[derive(Debug, Clone)]
pub struct Genome {
    /// All connections of the genome including connections from
    /// and to input and output neurons and bias neuron.
    /// input_id, output_id, innovation_number.
    /// Ordered by innovation number!
    graph: Graph<(), Connection>,
    /// The indices of the special nodes
    bias: NodeIndex,
    inputs: Vec<NodeIndex>,
    outputs: Vec<NodeIndex>,
    // The connections of the graph in ascending innovation order
    // // connections: Vec<&'a Connection>, // fixme: remove beacuse as we only ever add nodes probably petgraph edeglist will have the order.
}

impl PartialEq<Self> for Genome {
    // from https://github.com/petgraph/petgraph/issues/199#issuecomment-484077775
    fn eq(&self, other: &Self) -> bool {
        let a_ns = self.graph.raw_nodes().iter().map(|n| &n.weight);
        let b_ns = other.graph.raw_nodes().iter().map(|n| &n.weight);
        let a_es = self.graph.raw_edges().iter().map(|e| (e.source(), e.target(), &e.weight));
        let b_es = other.graph.raw_edges().iter().map(|e| (e.source(), e.target(), &e.weight));
        a_ns.eq(b_ns) && a_es.eq(b_es)
    }
}


impl Genome {
    /// Build a fully connected initial (default) genome.
    pub fn new(inputs: usize, outputs: usize) -> Genome {
        let mut graph = Graph::default();
        let bias: NodeIndex = graph.add_node(());
        let inputs: Vec<NodeIndex> = (0..inputs).map(|i| graph.add_node(())).collect();
        let outputs: Vec<NodeIndex> = (0..outputs).map(|i| graph.add_node(())).collect();

        let mut c = 0;
        for o in outputs.iter() {
            graph.add_edge(bias, *o, Connection { weight: 1., enabled: true, innovation_id: c });
            c = c + 1;
            for i in inputs.iter() {
                graph.add_edge(*i, *o, Connection { weight: 1., enabled: true, innovation_id: c });
                c = c + 1;
            }
        }

        Genome { graph, bias, inputs, outputs }
    }

    /// Vector of input node indices
    pub fn inputs(&self) -> &'_ Vec<NodeIndex> { &self.inputs }

    /// Vector of output node indices
    pub fn outputs(&self) -> &'_ Vec<NodeIndex> { &self.outputs }

    /// Bias node index
    pub fn bias(&self) -> NodeIndex { self.bias }

    /// Get the connection between given source and target neuron.
    pub fn connection(&self, source: NodeIndex, target: NodeIndex) -> Option<&'_ Connection> {
        assert_eq!(self.graph.edges_connecting(source, target).count(), 1);
        self.graph.edges_connecting(source, target).next().map(|er| er.weight())
    }

    /// Insert a new connection between the given two nodes.
    ///  - There must not already be a connection.
    ///  - Passed innovation id must be greater than any innovation id in this genome.
    pub fn insert(&mut self, source: NodeIndex, target: NodeIndex, innovation_id: usize) -> EdgeIndex {
        assert!(innovation_id > self.graph.edge_weights().last().unwrap().innovation_id);
        assert_eq!(self.graph.edges_connecting(source, target).count(), 0);
        // fixme: random weight?
        self.graph.add_edge(source, target, Connection { enabled: true, weight: 0., innovation_id })
    }

    /// Split an existing connection into two by inserting a new node and disabling the present connection.
    /// - There must be a connection from source to target
    /// - The connection from source to target must be enabled
    pub fn split(&mut self, source: NodeIndex, target: NodeIndex, innovation_id: usize) -> (EdgeIndex, EdgeIndex) {
        assert_eq!(self.graph.edges_connecting(source, target).count(), 1);
        assert_eq!(self.graph.edges_connecting(source, target).next().unwrap().weight().enabled, true);
        let old = self.graph.edges_connecting_mut(source, target).next().unwrap().weight();
        old.enabled = false;

        let node = self.graph.add_node(());

        let e1 = self.graph.add_edge(source, node, Connection {
            weight: 0.0, // fixme: weight
            enabled: true,
            innovation_id: innovation_id + 0,
        });

        let e2 = self.graph.add_edge(source, node, Connection {
            weight: 0.0, // fixme: weight
            enabled: true,
            innovation_id: innovation_id + 1,
        });

        (e1, e2)
    }

    /// Get the connection with given innovation number.
    pub fn innovation(&self, id: usize) -> Option<&'_ Connection> {
        if id > self.graph.edge_weights().last().unwrap().innovation_id {
            return None;
        }
        unimplemented!()
        // match self.connections.binary_search_by_key(&id, move |conn| conn.innovation_id) {
        //     Ok(pos) => Some(&self.connections[pos]),
        //     Err(_) => None
        // }
    }

    /// > Genes that do not match are either disjoint or excess, depending on whether they occur
    /// > within or outside the range of the other parent’s innovation
    /// > numbers. They represent structure that is not present in the other genome.
    /// [Pag. 110, NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
    pub fn excess(&self, other: &Genome) -> impl Iterator<Item=&'_ Connection> {
        let threshold = other.graph.edge_weights().last().unwrap().innovation_id;
        self.graph.edge_weights()
            .filter(move |&gene| gene.innovation_id > threshold)
    }

    pub fn disjoint<'a>(&'a self, other: &'a Genome) -> impl Iterator<Item=&'a Connection> {
        let threshold = other.graph.edge_weights().last().unwrap().innovation_id;
        self.mismatching(other)
            .filter(move |&gene| gene.innovation_id < threshold)
    }

    /// An iterator over connections that are in self, but not in other.
    pub fn mismatching<'a>(&'a self, other: &'a Genome) -> impl Iterator<Item=&'a Connection> {
        Gn::new_scoped(move |mut s| {
            for e1 in self.graph.edge_weights() {
                // fixme eventually performance optimize this
                let mut found = false;
                for e2 in other.graph.edge_weights() {
                    if e1.innovation_id == e2.innovation_id {
                        found = true;
                        break;
                    }
                }
                if !found {
                    s.yield_(e1);
                }
            }
            done!();
        })
    }

    /// An iterator over the aligned connections of the two genomes. The first element of the tuples
    /// will be the connection of self, the second element the connection of other.
    pub fn matching<'o>(&'o self, other: &'o Genome) -> impl Iterator<Item=(&'o Connection, &'o Connection)> {
        Gn::new_scoped(move |mut s| {
            for e1 in self.graph.edge_weights() {
                for e2 in other.graph.edge_weights() {
                    if e1.innovation_id == e2.innovation_id {
                        s.yield_((e1, e2));
                    }
                }
            }
            done!();
            // fixme eventually performance optimize this
            // while true {
            //     let e1 = it1.next();
            //     let e2 = it2.next();
            //     if e1.is_none() || e2.is_none() {
            //         break;
            //     }
            //     let e1 = e1.unwrap();
            //     let e2 = e2.unwrap();
            //     if e1.innovation_id == e2.innovation_id {
            //         s.yield_((e1, e2));
            //     }
            //     if e1.innovation_id >= e2.innovation_id {
            //         e2 = it2.next();
            //     } else {
            //         e1 = it1.next();
            //     }
            // }
            // done!();
        })
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
        let N = max(self.graph.edge_count(), other.graph.edge_count());

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
            genomes: vec![Genome::new(inputs, outputs); size],
            n_inputs: inputs,
            n_outputs: outputs,
            //fixme: representative could also be a reference or an index into the genomes array?!
            species: vec![Species { representative: Genome::new(inputs, outputs) }],
        }
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
    fn new_genome_should_be_fully_connected() {
        let g = Genome::new(3, 3);
        assert_eq!(g.graph.edge_count(), 3 * 3 + 3);

        assert!(g.connection(g.bias, g.outputs[0]).is_some());
        for i in 0..3 {
            assert!(g.connection(g.inputs[0], g.outputs[0]).is_some());
        }
    }

    #[test]
    fn two_new_new_genomes_should_be_fully_aligned() {
        let g1 = Genome::new(3, 3);
        let g2 = Genome::new(3, 3);

        let overlap1: Vec<(&Connection, &Connection)> = g1.matching(&g2).collect();
        let overlap2: Vec<(&Connection, &Connection)> = g2.matching(&g1).collect();

        for e in &overlap1 {
            println!("{:?}", &e);
        }
        assert_eq!(overlap1.len(), 12);
        assert_eq!(overlap2.len(), 12);
        assert_eq!(overlap1, overlap2);

        let excess1: Vec<&Connection> = g1.excess(&g2).collect();
        let disjoint1: Vec<&Connection> = g1.disjoint(&g2).collect();

        let excess2: Vec<&Connection> = g2.excess(&g1).collect();
        let disjoint2: Vec<&Connection> = g2.disjoint(&g1).collect();

        assert_eq!(excess1.len(), 0);
        assert_eq!(disjoint1.len(), 0);
        assert_eq!(excess2.len(), 0);
        assert_eq!(disjoint2.len(), 0);
    }

    #[test]
    fn genome_matching_with_one_excess() {
        /// Create a graph that has one extra hidden node between
        /// bias and output0 nodes.
        fn initial_genome() -> Genome {
            let mut g1 = Genome::new(2, 2);
            let edges = g1.split(g1.bias, g1.outputs[0], 12);
            g1
        }

        let mut g1 = initial_genome();
        let mut g2 = initial_genome();
        g2.insert()

        let overlap1: Vec<(&Connection, &Connection)> = g1.matching(&g2).collect();
        let overlap2: Vec<(&Connection, &Connection)> = g2.matching(&g1).collect();

        assert_eq!(overlap1.len(), 12);
        assert_eq!(overlap2.len(), 12);
        assert_eq!(overlap1, overlap2);

        let excess1: Vec<&Connection> = g1.excess(&g2).collect();
        let disjoint1: Vec<&Connection> = g1.disjoint(&g2).collect();

        let excess2: Vec<&Connection> = g2.excess(&g1).collect();
        let disjoint2: Vec<&Connection> = g2.disjoint(&g1).collect();

        assert_eq!(excess1.len(), 2);
        assert_eq!(disjoint1.len(), 0);
        assert_eq!(excess2.len(), 0);
        assert_eq!(disjoint2.len(), 0);
    }

    #[test]
    fn new_population_should_contain_one_species() {
        let p = Population::new(10, 3, 3);
        assert_eq!(p.species().len(), 1);
    }
}
