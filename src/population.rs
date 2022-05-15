use generator::Gn;
use ndarray::prelude::*;
use std::cmp::max;
use petgraph::{Graph};
use petgraph::graph::{NodeIndex, EdgeIndex};
use std::sync::Arc;
use std::cell::RefCell;

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

/// Uniquely identify a node within all genomes of a population.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct NodeId(NodeIndex);

/// Uniquely identify an edge within all genomes of a population.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct EdgeId(EdgeIndex);


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
    graph: Graph<NodeId, (EdgeId, Connection)>,
    /// The indices of the special nodes
    bias: NodeIndex,
    inputs: Vec<NodeIndex>,
    outputs: Vec<NodeIndex>,
    // The connections of the graph in ascending innovation order
    // // connections: Vec<&'a Connection>, // fixme: remove beacuse as we only ever add nodes probably petgraph edeglist will have the order.
}

/// Track the overlay graph of a set of genomes to provide unique node and edge ids.
pub struct GenomePool {
    overlay: Graph<(), ()>,
    genomes: Vec<Genome>,
}


impl GenomePool {
    /// Generate an empty pool for genomes with the given number of inputs and outputs.
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let mut graph: Graph<(), ()> = Graph::default();
        let bias: NodeIndex = graph.add_node(());
        let inputs: Vec<NodeIndex> = (0..inputs).map(|i| graph.add_node(())).collect();
        let outputs: Vec<NodeIndex> = (0..outputs).map(|i| graph.add_node(())).collect();

        for o in outputs.iter() {
            graph.add_edge(bias, *o, ());
            for i in inputs.iter() {
                graph.add_edge(*i, *o, ());
            }
        }

        Self { overlay: graph, genomes: Vec::new() }
    }

    pub fn spawn_genome(&self) -> Genome {
        Genome {
            graph: Default::default(),
            bias: Default::default(),
            inputs: vec![],
            outputs: vec![],
        }
    }

    /// Get the edge of the overlay from source to target.
    pub fn find_edge(&self, source: &NodeId, target: &NodeId) -> Option<EdgeId> {
        self.overlay.find_edge(source.0, target.0).map(|e| EdgeId(e))
    }

    /// Insert a new edge between given nodes. Returns an existing edge id if already present.
    /// TODO: Within the same generation this might be a idempotent operation, whereas a new edge between
    ///       the same nodes may be inserted for different generations.
    pub fn insert_or_retrieve(&mut self, source: &NodeId, target: &NodeId) -> EdgeId {
        let e = self.overlay.find_edge(source.0, target.0);
        match e {
            Some(e) => { EdgeId(e) }
            None => {
                EdgeId(self.overlay.add_edge(source.0, target.0, ()))
            }
        }
    }

    /// Split the edge between two existing nodes adding a new node and two new edges.
    /// Panics if there is no edge between the two nodes.
    /// Return the id of the new node and the two edges (from source to new node and new node to
    /// target).
    pub fn split(&mut self, source: &NodeId, target: &NodeId) -> (NodeId, (EdgeId, EdgeId)) {
        match self.overlay.find_edge(source.0, target.0) {
            None => { panic!("Edge not found"); }
            Some(e) => {
                let node = self.overlay.add_node(());
                let e1 = self.overlay.add_edge(source.0, node, ());
                let e2 = self.overlay.add_edge(node, target.0, ());
                (NodeId(node), (EdgeId(e1), EdgeId(e2)))
            }
        }
    }
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
    // /// Build a fully connected initial (default) genome.
    // pub fn new(inputs: usize, outputs: usize) -> Genome {}

    /// Vector of input node indices
    pub fn inputs(&self) -> &'_ Vec<NodeIndex> { &self.inputs }

    /// Vector of output node indices
    pub fn outputs(&self) -> &'_ Vec<NodeIndex> { &self.outputs }

    /// Bias node index
    pub fn bias(&self) -> NodeIndex { self.bias }

    /// Get the connection between given source and target neuron.
    pub fn connection(&self, source: NodeIndex, target: NodeIndex) -> Option<&'_ (EdgeId, Connection)> {
        assert_eq!(self.graph.edges_connecting(source, target).count(), 1);
        self.graph.edges_connecting(source, target).next().map(|er| er.weight())
    }

    /// Insert a new connection between the given two nodes.
    ///  - Panics if source or target are not in the graph or the overlay.
    ///  - There must not already be a connection.
    ///  - Passed innovation id must be greater than any innovation id in this genome.
    pub fn insert(&mut self, source: NodeIndex, target: NodeIndex, overlay: &mut GenomePool) {
        assert_eq!(self.graph.edges_connecting(source, target).count(), 0);
        // check if there is already a matching connection in the overlay
        let source_id = self.graph.node_weight(source).unwrap();
        let target_id = self.graph.node_weight(source).unwrap();

        // retrieve existing or add new edge
        let edge_id = overlay.insert_or_retrieve(source_id, target_id);
        self.graph.add_edge(source, target, (edge_id, Connection { enabled: true, weight: 0., innovation_id: 0 }));
    }

    /// Split an existing connection into two by inserting a new node and disabling the present connection.
    /// - Panics if there is no connection from source to target.
    /// - Panic if the connection from source to target is disabled.
    pub fn split(&mut self, source: NodeIndex, target: NodeIndex, overlay: &mut GenomePool) -> ((NodeIndex, NodeId), ((EdgeIndex, EdgeId), (EdgeIndex, EdgeId))) {
        if !self.graph.edges_connecting(source, target).next().unwrap().weight().1.enabled {
            panic!("Edge from {source:?} to {target:?} is disabled");
        }
        // check if there is already a matching connection in the overlay
        let source_id = self.graph.node_weight(source).unwrap();
        let target_id = self.graph.node_weight(source).unwrap();

        let (node_id, (head_id, tail_id)) = overlay.split(source_id, target_id);

        let old = self.graph.find_edge(source, target).unwrap();
        let (_, a) = self.graph.edge_weight_mut(old).unwrap();
        a.enabled = false;

        let node = self.graph.add_node(node_id.clone());

        let e1 = self.graph.add_edge(source, node, (head_id.clone(), Connection {
            weight: 0.0, // fixme: weight
            enabled: true,
            innovation_id: 0,
        }));

        let e2 = self.graph.add_edge(node, target, (tail_id.clone(), Connection {
            weight: 0.0, // fixme: weight
            enabled: true,
            innovation_id: 0,
        }));
        ((node, node_id), ((e1, head_id), (e2, tail_id)))
    }

    // /// Get the connection with given innovation number.
    // pub fn innovation(&self, id: usize) -> Option<&'_ Connection> {
    //     if id > self.graph.edge_weights().last().unwrap().innovation_id {
    //         return None;
    //     }
    //     unimplemented!()
    //     // match self.connections.binary_search_by_key(&id, move |conn| conn.innovation_id) {
    //     //     Ok(pos) => Some(&self.connections[pos]),
    //     //     Err(_) => None
    //     // }
    // }

    /// > Genes that do not match are either disjoint or excess, depending on whether they occur
    /// > within or outside the range of the other parent’s innovation
    /// > numbers. They represent structure that is not present in the other genome.
    /// [Pag. 110, NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
    pub fn excess<'a>(&'a self, other: &'a Genome) -> impl Iterator<Item=&'a (EdgeId, Connection)> {
        let threshold = &other.graph.edge_weights().last().unwrap().0;
        self.graph.edge_weights()
            .filter(move |&gene| &gene.0 > threshold)
    }

    pub fn disjoint<'a>(&'a self, other: &'a Genome) -> impl Iterator<Item=&'a (EdgeId, Connection)> {
        let threshold = &other.graph.edge_weights().last().unwrap().0;
        self.mismatching(other)
            .filter(move |&gene| &gene.0 < threshold)
    }

    /// An iterator over connections that are in self, but not in other.
    pub fn mismatching<'a>(&'a self, other: &'a Genome) -> impl Iterator<Item=&'a (EdgeId, Connection)> {
        Gn::new_scoped(move |mut s| {
            for e1 in self.graph.edge_weights() {
                // fixme eventually performance optimize this
                let mut found = false;
                for e2 in other.graph.edge_weights() {
                    if e1.0 == e2.0 {
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
    pub fn matching<'o>(&'o self, other: &'o Genome) -> impl Iterator<Item=(&'o (EdgeId, Connection), &'o (EdgeId, Connection))> {
        Gn::new_scoped(move |mut s| {
            for e1 in self.graph.edge_weights() {
                for e2 in other.graph.edge_weights() {
                    if e1.0 == e2.0 {
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
        let avg_difference = 0.; // fixme implement average weight difference of matching genes

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

/// A collection of genomes
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

    /// Allow iteration over all species within the population.
    pub fn species(&self) -> &'_ Vec<Species> {
        &self.species
    }

    /// Allow iteration over the genomes of all individuals of the population.
    pub fn genomes(&self) -> &'_ Vec<Genome> {
        &self.genomes
    }

    /// The total number of individuals in the population.
    pub fn len(&self) -> usize {
        self.genomes.len()
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

/// Create offspring by mutation and mating. May create new species.
/// returns the next generation of the population.
pub fn evolve(population: &Population, fitness: &Array1<f32>) -> Population {
    let average_fitness = fitness.sum() / fitness.len() as f32;
    let n_organisms = population.len();

    let species = population.species();
    let genomes = population.genomes();

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

    for specie in species {
        let specie_fitness = specie.calculate_average_fitness();
        let offspring_size = if total_average_fitness <= 0f64 {
            specie.organisms.len()
        } else {
            (specie_fitness * organisms_by_average_fitness).round() as usize
        };
        if offspring_size > 0 {
            specie.generate_offspring(offspring_size, &organisms);
        } else {
            specie.remove_organisms();
        }
    }

    // fixme build the next generation
    let next_generation: Vec<Genome> = vec![];

    Population { n_outputs: self.n_outputs, n_inputs: self.n_inputs, genomes: next_generation, species: vec![] }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn mutation_tracker_should_generate_ascending_numbers() {
    //     let mut t = Arc::<RefCell<usize>>::new(RefCell::new(0));
    //     assert_eq!(t.next_id(), 1);
    //     assert_eq!(t.next_id(), 2);
    // }

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

        let overlap1: Vec<(&(EdgeId, Connection), &(EdgeId, Connection))> = g1.matching(&g2).collect();
        let overlap2: Vec<(&(EdgeId, Connection), &(EdgeId, Connection))> = g2.matching(&g1).collect();

        for e in &overlap1 {
            println!("{:?}", &e);
        }
        assert_eq!(overlap1.len(), 12);
        assert_eq!(overlap2.len(), 12);
        assert_eq!(overlap1, overlap2);

        let excess1: Vec<&(EdgeId, Connection)> = g1.excess(&g2).collect();
        let disjoint1: Vec<&(EdgeId, Connection)> = g1.disjoint(&g2).collect();

        let excess2: Vec<&(EdgeId, Connection)> = g2.excess(&g1).collect();
        let disjoint2: Vec<&(EdgeId, Connection)> = g2.disjoint(&g1).collect();

        assert_eq!(excess1.len(), 0);
        assert_eq!(disjoint1.len(), 0);
        assert_eq!(excess2.len(), 0);
        assert_eq!(disjoint2.len(), 0);
    }

    #[test]
    fn test_split() {
        let mut pool = GenomePool::new(2, 2);
        let mut g = pool.initial_genome();
        assert_eq!(g.graph.edge_count(), 6);
        let (node, edges) = g.split(g.bias, g.outputs[0], &mut pool);
        assert_eq!(g.graph.node_count(), 6);
        assert_eq!(g.graph.edge_count(), 8);
        assert!(!g.graph.edges_connecting(g.bias, g.outputs[0]).next().unwrap().weight().1.enabled);
        assert!(g.graph.edge_weight(edges.0.0).unwrap().1.enabled);
        assert!(g.graph.edge_weight(edges.1.0).unwrap().1.enabled);
    }

    #[test]
    fn test_insert() {
        let mut overlay = GenomePool::new();
        let mut g = Genome::new(2, 2, &mut overlay);
        assert_eq!(g.graph.edge_count(), 6);
        let (node, edges) = g.split(g.bias, g.outputs[0], &mut overlay);
        assert_eq!(g.graph.node_count(), 6);
        assert_eq!(g.graph.edge_count(), 8);
        g.insert(node.0, g.outputs[1], &mut overlay);
        assert!(!g.graph.edges_connecting(g.bias, g.outputs[0]).next().unwrap().weight().1.enabled);
        assert!(g.graph.edge_weight(edges.0.0).unwrap().1.enabled);
        assert!(g.graph.edge_weight(edges.1.0).unwrap().1.enabled);
    }

    // #[test]
    // fn genome_matching_with_one_excess() {
    //     /// Create a graph that has one extra hidden node between
    //     /// bias and output0 nodes.
    //     fn initial_genome() -> (NodeIndex, Genome) {
    //         let mut g1 = Genome::new(2, 2);
    //         let (node, edges) = g1.split(g1.bias, g1.outputs[0], 12);
    //         (node, g1)
    //     }
    //
    //     let (_, mut g1) = initial_genome();
    //     let (node, mut g2) = initial_genome();
    //     let foo = g2.insert(node, g2.outputs[0], 14);
    //
    //     let overlap1: Vec<(&Connection, &Connection)> = g1.matching(&g2).collect();
    //     let overlap2: Vec<(&Connection, &Connection)> = g2.matching(&g1).collect();
    //
    //     assert_eq!(overlap1.len(), 8);
    //     assert_eq!(overlap2.len(), 8);
    //     assert_eq!(overlap1, overlap2);
    //
    //     let excess1: Vec<&Connection> = g1.excess(&g2).collect();
    //     let disjoint1: Vec<&Connection> = g1.disjoint(&g2).collect();
    //
    //     let excess2: Vec<&Connection> = g2.excess(&g1).collect();
    //     let disjoint2: Vec<&Connection> = g2.disjoint(&g1).collect();
    //
    //     assert_eq!(excess1.len(), 2);
    //     assert_eq!(disjoint1.len(), 0);
    //     assert_eq!(excess2.len(), 0);
    //     assert_eq!(disjoint2.len(), 0);
    // }

    #[test]
    fn new_population_should_contain_one_species() {
        let p = Population::new(10, 3, 3);
        assert_eq!(p.species().len(), 1);
    }
}
