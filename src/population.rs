use generator::{Gn, Generator};
use ndarray::prelude::*;
use std::cmp::max;
use petgraph::{Graph};
use petgraph::graph::{NodeIndex, EdgeIndex};
use std::sync::Arc;
use std::cell::RefCell;

/// A connection Gene
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ConnectionGene {
    /// The connection strength
    pub weight: f64,
    /// Whether the connection is enabled or not
    pub enabled: bool,
}

impl Default for ConnectionGene {
    fn default() -> Self {
        Self { weight: 1., enabled: true }
    }
}

/// Uniquely identify a node within all genomes of a population.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct NodeId(NodeIndex);

/// Uniquely identify an edge within all genomes of a population.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct EdgeId(EdgeIndex);

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Connection {
    pub id: EdgeId,
    pub gene: ConnectionGene,
}

impl Connection {
    pub fn new(id: usize, gene: ConnectionGene) -> Self {
        Self { id: EdgeId(EdgeIndex::new(id)), gene }
    }
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
    pub(crate) graph: Graph<NodeId, Connection>,
    /// The indices of the special nodes
    pub(crate) bias: NodeIndex,
    pub(crate) inputs: Vec<NodeIndex>,
    pub(crate) outputs: Vec<NodeIndex>,
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
    /// Build a fully connected initial (default) genome.
    pub fn new(inputs: usize, outputs: usize) -> Genome {
        let mut graph: Graph<NodeId, Connection> = Graph::default();
        let bias: NodeIndex = graph.add_node(NodeId(NodeIndex::new(0)));
        let inputs: Vec<NodeIndex> = (0..inputs).map(|i| graph.add_node(NodeId(NodeIndex::new(i + 1)))).collect();
        let outputs: Vec<NodeIndex> = (0..outputs).map(|i| graph.add_node(NodeId(NodeIndex::new(inputs.len() + i + 1)))).collect();

        let mut c: usize = 0;
        for o in outputs.iter() {
            graph.add_edge(bias, *o, Connection::new(c, ConnectionGene::default()));
            c = c + 1;
            for i in inputs.iter() {
                graph.add_edge(*i, *o, Connection::new(c, ConnectionGene::default()));
                c = c + 1;
            }
        }

        Self { graph, bias, inputs, outputs }
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
        self.graph.add_edge(source, target, Connection { id: edge_id, gene: ConnectionGene { enabled: true, weight: 0. } });
    }

    /// Split an existing connection into two by inserting a new node and disabling the present connection.
    /// - Panics if there is no connection from source to target.
    /// - Panic if the connection from source to target is disabled.
    pub fn split(&mut self, source: NodeIndex, target: NodeIndex, overlay: &mut GenomePool) -> ((NodeIndex, NodeId), ((EdgeIndex, EdgeId), (EdgeIndex, EdgeId))) {
        if !self.graph.edges_connecting(source, target).next().unwrap().weight().gene.enabled {
            panic!("Edge from {source:?} to {target:?} is disabled");
        }
        // check if there is already a matching connection in the overlay
        let source_id = self.graph.node_weight(source).unwrap();
        let target_id = self.graph.node_weight(source).unwrap();

        let (node_id, (head_id, tail_id)) = overlay.split(source_id, target_id);

        let old = self.graph.find_edge(source, target).unwrap();
        self.graph.edge_weight_mut(old).unwrap().gene.enabled = false;

        let node = self.graph.add_node(node_id.clone());

        let e1 = self.graph.add_edge(source, node, Connection {
            id: head_id.clone(),
            gene: ConnectionGene {
                weight: 0.0, // fixme: weight
                enabled: true,
            },
        });

        let e2 = self.graph.add_edge(node, target, Connection {
            id: tail_id.clone(),
            gene: ConnectionGene {
                weight: 0.0, // fixme: weight
                enabled: true,
            },
        });
        ((node, node_id), ((e1, head_id), (e2, tail_id)))
    }

    /// Get the connection with given innovation number.
    pub fn innovation(&self, id: usize) -> Option<&'_ Connection> {
        if id > self.graph.edge_weights().last().unwrap().id.0.index() {
            return None;
        }
        unimplemented!()
        // match self.connections.binary_search_by_key(&id, move |conn| conn.innovation_id) {
        //     Ok(pos) => Some(&self.connections[pos]),
        //     Err(_) => None
        // }
    }

    /// Iterate over the aligned parts of two genomes.
    /// Visually one can represent the matching (M), disjoint (D) and excess (E) genes as follows:
    /// Gene Position | 0 2 4 .......  N
    /// Genome 1      | ## #####  ## ###
    /// Genome 2      | # ###   ######
    /// Kind          | MDDMMDDDDDMMDMEE
    /// The returned tuple contains:
    ///  - a bool indicating whether the current items is an excess gene.
    ///    This means that either one of the returned options is and will remain None for the rest of the iterator.
    ///  - The optional matching gene of a
    ///  - The optional matching gene of b.
    /// If either one of the optionals is set and the bool is false, it is a disjoint gene.
    /// If both options as set, then it is a matching gene.
    ///
    pub fn zip<'a>(a: &'a Genome, b: &'a Genome) -> impl Iterator<Item=(bool, Option<&'a Connection>, Option<&'a Connection>)> {
        Gn::new_scoped(move |mut s| {
            let mut iter_a = a.graph.edge_weights().into_iter().fuse();
            let mut iter_b = b.graph.edge_weights().into_iter().fuse();

            let mut cur_a = iter_a.next();
            let mut cur_b = iter_b.next();

            // loop until both iterators are exhausted
            while cur_a.is_some() || cur_b.is_some() {
                // have we reached the end of one of the iterators?
                match (cur_a, cur_b) {
                    // none of the two iterators is exhausted
                    (Some(a), Some(b)) => {
                        // match, advance both iterators.
                        if a.id == b.id {
                            s.yield_((false, Some(a), Some(b)));
                            cur_a = iter_a.next();
                            cur_b = iter_b.next();
                        } else if a.id > b.id {
                            // we have a (disjoint) gene in b that is not in a
                            s.yield_((false, None, Some(b)));
                            cur_b = iter_b.next();
                        } else {
                            // we have a (disjoint) gene in a that is not in b
                            s.yield_((false, Some(a), None));
                            cur_a = iter_a.next();
                        }
                    }
                    // one of the two iterators is exhausted
                    // It does not matter if we advance the exhausted iterator
                    // as it is fused and will continue yielding None once exhausted.
                    _ => {
                        s.yield_((true, cur_a, cur_b));
                        cur_a = iter_a.next();
                        cur_b = iter_b.next();
                    }
                }
            }
            done!();
        })
    }

    /// > Genes that do not match are either disjoint or excess, depending on whether they occur
    /// > within or outside the range of the other parent’s innovation
    /// > numbers. They represent structure that is not present in the other genome.
    /// [Pag. 110, NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
    pub fn excess<'a>(&'a self, other: &'a Genome) -> impl Iterator<Item=&'a Connection> {
        let threshold = &other.graph.edge_weights().last().unwrap().id;
        self.graph.edge_weights()
            .filter(move |&gene| &gene.id > threshold)
    }

    pub fn disjoint<'a>(&'a self, other: &'a Genome) -> impl Iterator<Item=&'a Connection> {
        let threshold = &other.graph.edge_weights().last().unwrap().id;
        self.mismatching(other)
            .filter(move |&gene| &gene.id < threshold)
    }

    /// An iterator over connections that are in self, but not in other.
    pub fn mismatching<'a>(&'a self, other: &'a Genome) -> impl Iterator<Item=&'a Connection> {
        Genome::zip(self, other)
            .filter(|(_, s, o)| s.is_some() && o.is_none())
            .map(|(_, s, _)| s.unwrap())
    }

    /// An iterator over the aligned connections of the two genomes. The first element of the tuples
    /// will be the connection of self, the second element the connection of other.
    pub fn matching<'o>(&'o self, other: &'o Genome) -> impl Iterator<Item=(&'o Connection, &'o Connection)> {
        Genome::zip(self, other)
            .filter(|(_, a, b)| a.is_some() && b.is_some())
            .map(|(_, a, b)| (a.unwrap(), b.unwrap()))
    }
}

/// A species owns its genomes
#[derive(Debug)]
pub struct Species {
    /// The representative defines whether another
    /// individual belongs to the species or not.
    pub(crate) representative: Genome,
    pub(crate) id: usize,
    pub(crate) genomes: Vec<Genome>,
}

impl Species {
    pub fn len(&self) -> usize { self.genomes.len() }
    pub fn iter(&self) -> impl Iterator<Item=&'_ Genome> {
        self.genomes.iter()
    }
}

// pub enum Mutation {
//     AddEdge { source_id: int, target_id: int },
//     AddNode(),
//     ChangeWeight(),
//     ToggleExpression(),
// }

/// A population owns its species and hence all genomes.
#[derive(Debug)]
pub struct Population {
    /// The number of input or sensor neurons.
    pub(crate) n_inputs: usize,
    /// The number of output  neurons.
    pub(crate) n_outputs: usize,
    /// The species of the population.
    pub(crate) species: Vec<Species>,
    // champion_fitness: f64,
    // epochs_without_improvements: usize,
    // /// champion of the population
    // pub champion: Option<Organism>,
}

impl Population {
    pub fn new(individuals: usize, inputs: usize, outputs: usize) -> Population {
        Population {
            n_inputs: inputs,
            n_outputs: outputs,
            species: vec![
                Species {
                    representative: Genome::new(inputs, outputs),
                    id: 0,
                    genomes: vec![Genome::new(inputs, outputs); individuals],
                }
            ],
        }
    }

    /// Allow iteration over all species within the population.
    pub fn species(&self) -> &'_ Vec<Species> {
        &self.species
    }

    /// Allow iteration over the genomes of all individuals of the population.
    pub fn genomes(&self) -> impl Iterator<Item=&'_ Genome> {
        self.species.iter().map(|s| s.genomes.iter()).flatten()
    }

    /// The total number of individuals in the population.
    pub fn len(&self) -> usize {
        self.genomes().count()
    }

    pub fn iter(&self) -> Generator<'_, (), (&'_ Species, &'_ Genome)> {// impl Iterator<Item=> {
        let g = Gn::new_scoped(move |mut s| {
            for species in &self.species {
                for genome in &species.genomes {
                    s.yield_((species, genome));
                }
            }
            done!();
        });
        g
    }
}

// fixme: By convention we should also implement this trait with a simple delegation to
//        population.iter()
impl<'a> IntoIterator for &'a Population {
    type Item = (&'a Species, &'a Genome);
    type IntoIter = Generator<'a, (), Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
        // Self::IntoIter {
        //     population: self,
        //     iter: self.species.iter().map(|s|s.genomes.iter()),
        // }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zipping_genomes_yields_correct_results() {
        {
            let a = Genome::new(2, 2);
            let b = Genome::new(2, 2);
            let zipped_1: Vec<(bool, Option<&Connection>, Option<&Connection>)> = Genome::zip(&a, &b).collect();
            let zipped_2: Vec<(bool, Option<&Connection>, Option<&Connection>)> = Genome::zip(&a, &b).collect();
            assert_eq!(zipped_1.len(), 6);
            assert_eq!(zipped_1, zipped_2);
        }
        {
            /// Setup, 2 inputs, 2, outputs, one bias -> 6 matching base genes!
            /// Gene Position | 0 2 4 6 8 .... N=15
            /// Genome A      | ####### # ## ###
            /// Genome B      | ###### ######
            /// Type          | MMMMMMDDMDMMDEEE
            let mut a = Genome::new(2, 2);
            let mut b = Genome::new(2, 2);
            let n6 = a.graph.add_node(NodeId(NodeIndex::new(6)));
            let n7 = a.graph.add_node(NodeId(NodeIndex::new(7)));
            let n8 = a.graph.add_node(NodeId(NodeIndex::new(8)));
            let n6 = b.graph.add_node(NodeId(NodeIndex::new(6)));
            let n7 = b.graph.add_node(NodeId(NodeIndex::new(7)));
            let n8 = b.graph.add_node(NodeId(NodeIndex::new(8)));

            // position 6, disjoint only in a
            a.graph.add_edge(a.bias, n6, Connection::new(6, ConnectionGene::default()));
            // position 7, disjoint only in b
            b.graph.add_edge(b.inputs[0], n6, Connection::new(7, ConnectionGene::default()));
            // position 8, matching
            a.graph.add_edge(a.inputs[1], n7, Connection::new(8, ConnectionGene::default()));
            b.graph.add_edge(b.inputs[1], n7, Connection::new(8, ConnectionGene::default()));
            // position 9, disjoint only in b
            b.graph.add_edge(n6, b.outputs[0], Connection::new(9, ConnectionGene::default()));
            // position 10 & 11, matching
            a.graph.add_edge(a.inputs[0], n8, Connection::new(10, ConnectionGene::default()));
            b.graph.add_edge(b.inputs[0], n8, Connection::new(10, ConnectionGene::default()));
            a.graph.add_edge(n8, a.outputs[0], Connection::new(11, ConnectionGene::default()));
            b.graph.add_edge(n8, b.outputs[0], Connection::new(11, ConnectionGene::default()));
            // position 12, disjoint, only in b
            b.graph.add_edge(n6, b.outputs[0], Connection::new(12, ConnectionGene::default()));
            // position 13,14, & 15 excess of a
            a.graph.add_edge(n6, a.outputs[0], Connection::new(13, ConnectionGene::default()));
            a.graph.add_edge(a.inputs[1], n7, Connection::new(14, ConnectionGene::default()));
            a.graph.add_edge(n7, a.outputs[1], Connection::new(15, ConnectionGene::default()));


            let zipped_1: Vec<(bool, Option<&Connection>, Option<&Connection>)> = Genome::zip(&a, &b).collect();
            let zipped_2: Vec<(bool, Option<&Connection>, Option<&Connection>)> = Genome::zip(&a, &b).collect();

            assert_eq!(zipped_1.len(), 16);
            assert_eq!(zipped_1.len(), zipped_2.len());

            assert_eq!(a.matching(&b).count(), 9);
            assert_eq!(b.matching(&a).count(), 9);

            assert_eq!(a.mismatching(&b).count(), 4);
            assert_eq!(b.mismatching(&a).count(), 3);

            assert_eq!(a.excess(&b).count(), 3);
            assert_eq!(b.excess(&a).count(), 0);

            assert_eq!(a.disjoint(&b).count(), 1);
            assert_eq!(b.disjoint(&a).count(), 3);
        }
    }

    // #[test]
    // fn mutation_tracker_should_generate_ascending_numbers() {
    //     let mut t = Arc::<RefCell<usize>>::new(RefCell::new(0));
    //     assert_eq!(t.next_id(), 1);
    //     assert_eq!(t.next_id(), 2);
    // }

    // #[test]
    // fn new_genome_should_be_fully_connected() {
    //     let g = Genome::new(3, 3);
    //     assert_eq!(g.graph.edge_count(), 3 * 3 + 3);
    //
    //     assert!(g.connection(g.bias, g.outputs[0]).is_some());
    //     for i in 0..3 {
    //         assert!(g.connection(g.inputs[0], g.outputs[0]).is_some());
    //     }
    // }

    // #[test]
    // fn two_new_new_genomes_should_be_fully_aligned() {
    //     let g1 = Genome::new(3, 3);
    //     let g2 = Genome::new(3, 3);
    //
    //     let overlap1: Vec<(&(EdgeId, Connection), &(EdgeId, Connection))> = g1.matching(&g2).collect();
    //     let overlap2: Vec<(&(EdgeId, Connection), &(EdgeId, Connection))> = g2.matching(&g1).collect();
    //
    //     for e in &overlap1 {
    //         println!("{:?}", &e);
    //     }
    //     assert_eq!(overlap1.len(), 12);
    //     assert_eq!(overlap2.len(), 12);
    //     assert_eq!(overlap1, overlap2);
    //
    //     let excess1: Vec<&(EdgeId, Connection)> = g1.excess(&g2).collect();
    //     let disjoint1: Vec<&(EdgeId, Connection)> = g1.disjoint(&g2).collect();
    //
    //     let excess2: Vec<&(EdgeId, Connection)> = g2.excess(&g1).collect();
    //     let disjoint2: Vec<&(EdgeId, Connection)> = g2.disjoint(&g1).collect();
    //
    //     assert_eq!(excess1.len(), 0);
    //     assert_eq!(disjoint1.len(), 0);
    //     assert_eq!(excess2.len(), 0);
    //     assert_eq!(disjoint2.len(), 0);
    // }

    // #[test]
    // fn test_split() {
    //     let mut pool = GenomePool::new(2, 2);
    //     let mut g = pool.initial_genome();
    //     assert_eq!(g.graph.edge_count(), 6);
    //     let (node, edges) = g.split(g.bias, g.outputs[0], &mut pool);
    //     assert_eq!(g.graph.node_count(), 6);
    //     assert_eq!(g.graph.edge_count(), 8);
    //     assert!(!g.graph.edges_connecting(g.bias, g.outputs[0]).next().unwrap().weight().1.enabled);
    //     assert!(g.graph.edge_weight(edges.0.0).unwrap().1.enabled);
    //     assert!(g.graph.edge_weight(edges.1.0).unwrap().1.enabled);
    // }
    //
    // #[test]
    // fn test_insert() {
    //     let mut overlay = GenomePool::new();
    //     let mut g = Genome::new(2, 2, &mut overlay);
    //     assert_eq!(g.graph.edge_count(), 6);
    //     let (node, edges) = g.split(g.bias, g.outputs[0], &mut overlay);
    //     assert_eq!(g.graph.node_count(), 6);
    //     assert_eq!(g.graph.edge_count(), 8);
    //     g.insert(node.0, g.outputs[1], &mut overlay);
    //     assert!(!g.graph.edges_connecting(g.bias, g.outputs[0]).next().unwrap().weight().1.enabled);
    //     assert!(g.graph.edge_weight(edges.0.0).unwrap().1.enabled);
    //     assert!(g.graph.edge_weight(edges.1.0).unwrap().1.enabled);
    // }

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
