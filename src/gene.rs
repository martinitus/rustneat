extern crate rand;


//
// impl Eq for ConnectionGene {}
//
// impl PartialEq for ConnectionGene {
//     fn eq(&self, other: &ConnectionGene) -> bool {
//         self.source_id == other.source_id && self.target_id == other.target_id
//     }
// }
//
// impl Ord for ConnectionGene {
//     fn cmp(&self, other: &ConnectionGene) -> Ordering {
//         if self == other {
//             Ordering::Equal
//         } else if self.source_id == other.source_id {
//             if self.target_id > other.target_id {
//                 Ordering::Greater
//             } else {
//                 Ordering::Less
//             }
//         } else if self.source_id > other.source_id {
//             Ordering::Greater
//         } else {
//             Ordering::Less
//         }
//     }
// }
//
// impl PartialOrd for ConnectionGene {
//     fn partial_cmp(&self, other: &ConnectionGene) -> Option<Ordering> {
//         Some(self.cmp(other))
//     }
// }

impl ConnectionGene {
    /// Create a new gene with a specific connection
    pub fn new(source: usize, target: usize, innovation: usize) -> ConnectionGene {
        ConnectionGene {
            source_id: source,
            target_id: target,
            innovation_id: innovation,
            weight: 1.,
            enabled: true,
        }
    }

    /// Set gene enabled
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    /// Set gene disabled
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    /// Toggle the enable state
    pub fn toggle_enabled(&mut self) { self.enabled = !self.enabled; }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(n_in: usize, n_out: usize) -> ConnectionGene {
        ConnectionGene {
            source_id: n_in,
            target_id: n_out,
            ..ConnectionGene::default()
        }
    }

    #[test]
    fn should_be_able_to_binary_search_for_a_gene() {
        let mut genome = vec![g(0, 1), g(0, 2), g(3, 2), g(2, 3), g(1, 5)];
        genome.sort();
        genome.binary_search(&g(0, 1)).unwrap();
        genome.binary_search(&g(0, 2)).unwrap();
        genome.binary_search(&g(1, 5)).unwrap();
        genome.binary_search(&g(2, 3)).unwrap();
        genome.binary_search(&g(3, 2)).unwrap();
    }
}
