pub mod cluster_handle;
pub mod memory_cluster;

#[cfg(test)]
mod test {
    use crate::memory::cluster::memory_cluster::MemoryCluster;

    #[test]
    fn test_memory_cluster_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemoryCluster>();
        println!("MemoryCluster is Send + Sync");
    }
}
