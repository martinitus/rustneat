use population::{Connection, Genome};

/// Defines separation of genomes into species and possibility to mate genomes.
pub trait Compatibility {
    /// Compute the distance of two genomes. Higher distance means lower compatibility.
    fn distance(&self, genome1: &Genome, genome2: &Genome) -> f64;
}


/// δ = (c1 * E)/N + (c2 * D)/N + c3*W
#[derive(Debug)]
pub struct DefaultCompatibility {
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
}

impl Default for DefaultCompatibility {
    // From the original paper:
    // c3 was increased [for DPNV experiment which had population size of 1000 instead of 150]
    // to 3.0 in order to allow for finer distinctions between species based on weight
    // differences (the larger population has room for more species).
    fn default() -> Self {
        Self { c1: 1., c2: 1., c3: 0.4 }
    }
}


impl Compatibility for DefaultCompatibility {
    fn distance(&self, genome1: &Genome, genome2: &Genome) -> f64 {
        // Excess count, Disjoint count, and average weight difference of matching genes.
        let mut n_excess = 0;
        let mut n_disjoint = 0;
        let mut total_weight_distance: f64 = 0.;
        for (excess, e1, e2) in Genome::zip(genome1, genome2) {
            match (excess, e1, e2) {
                (true, _, _) => {
                    n_excess += 1
                }
                (false, Some(e1), Some(e2)) => {
                    total_weight_distance += f64::abs(e1.gene.weight - e2.gene.weight);
                }
                _ => {
                    n_disjoint += 1
                }
            }
        }

        // n, the number of genes in the larger genome, normalizes for genome size (n
        // can be set to 1 if both genomes are small, i.e., consist of fewer than 20 genes)
        let n = std::cmp::max(genome1.graph.edge_count(), genome2.graph.edge_count());

        return (self.c1 * n_excess as f64 + self.c2 * n_disjoint as f64 + self.c3 * total_weight_distance) / n as f64;
    }
}

