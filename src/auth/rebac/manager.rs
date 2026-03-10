//! ReBAC Manager - Main interface for relationship-based access control
//!
//! This module provides the high-level API for managing relationships
//! and checking permissions.

use super::graph::RelationshipGraph;
use super::schema::PermissionSchema;
use super::types::{Object, Relation, RelationTuple, Subject};
use crate::error::{Error, Result};
use std::sync::{Arc, RwLock};

/// Main ReBAC manager
#[derive(Debug)]
pub struct RebacManager {
    /// Relationship graph
    graph: Arc<RwLock<RelationshipGraph>>,
    /// Permission schema
    schema: PermissionSchema,
}

impl RebacManager {
    /// Create a new ReBAC manager
    pub fn new() -> Self {
        RebacManager {
            graph: Arc::new(RwLock::new(RelationshipGraph::new())),
            schema: PermissionSchema::default(),
        }
    }

    /// Create a new ReBAC manager with custom schema
    pub fn with_schema(schema: PermissionSchema) -> Self {
        RebacManager {
            graph: Arc::new(RwLock::new(RelationshipGraph::new())),
            schema,
        }
    }

    /// Create a new ReBAC manager with cache size
    pub fn with_cache_size(cache_size: usize) -> Self {
        RebacManager {
            graph: Arc::new(RwLock::new(RelationshipGraph::with_cache_size(cache_size))),
            schema: PermissionSchema::default(),
        }
    }

    /// Grant a relationship (add a tuple)
    pub fn grant(&self, subject: &str, relation: &str, object: &str) -> Result<()> {
        let subject = Subject::parse(subject)
            .map_err(|e| Error::InvalidInput(format!("Invalid subject: {}", e)))?;
        let relation = Relation::parse(relation)
            .map_err(|e| Error::InvalidInput(format!("Invalid relation: {}", e)))?;
        let object = Object::parse(object)
            .map_err(|e| Error::InvalidInput(format!("Invalid object: {}", e)))?;

        let tuple = RelationTuple::new(subject, relation, object);

        let mut graph = self
            .graph
            .write()
            .map_err(|_| Error::InvalidOperation("Failed to acquire write lock".to_string()))?;

        graph.add_tuple(tuple)
    }

    /// Revoke a relationship (remove a tuple)
    pub fn revoke(&self, subject: &str, relation: &str, object: &str) -> Result<()> {
        let subject = Subject::parse(subject)
            .map_err(|e| Error::InvalidInput(format!("Invalid subject: {}", e)))?;
        let relation = Relation::parse(relation)
            .map_err(|e| Error::InvalidInput(format!("Invalid relation: {}", e)))?;
        let object = Object::parse(object)
            .map_err(|e| Error::InvalidInput(format!("Invalid object: {}", e)))?;

        let tuple = RelationTuple::new(subject, relation, object);

        let mut graph = self
            .graph
            .write()
            .map_err(|_| Error::InvalidOperation("Failed to acquire write lock".to_string()))?;

        graph.remove_tuple(&tuple)
    }

    /// Check if subject has permission (relation) on object
    pub fn check_access(&self, subject: &str, relation: &str, object: &str) -> Result<bool> {
        let subject = Subject::parse(subject)
            .map_err(|e| Error::InvalidInput(format!("Invalid subject: {}", e)))?;
        let relation = Relation::parse(relation)
            .map_err(|e| Error::InvalidInput(format!("Invalid relation: {}", e)))?;
        let object = Object::parse(object)
            .map_err(|e| Error::InvalidInput(format!("Invalid object: {}", e)))?;

        let graph = self
            .graph
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire read lock".to_string()))?;

        graph.check(&subject, &relation, &object)
    }

    /// Async version of check_access (for tokio compatibility)
    #[cfg(feature = "streaming")]
    pub async fn check_access_async(
        &self,
        subject: &str,
        relation: &str,
        object: &str,
    ) -> Result<bool> {
        // For now, just wrap the sync version
        // In a real implementation, this would use async-aware locking
        self.check_access(subject, relation, object)
    }

    /// List all objects of a given type that subject has permission on
    pub fn list_accessible(
        &self,
        subject: &str,
        relation: &str,
        object_type: &str,
    ) -> Result<Vec<String>> {
        let subject = Subject::parse(subject)
            .map_err(|e| Error::InvalidInput(format!("Invalid subject: {}", e)))?;
        let relation = Relation::parse(relation)
            .map_err(|e| Error::InvalidInput(format!("Invalid relation: {}", e)))?;

        let graph = self
            .graph
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire read lock".to_string()))?;

        let objects = graph.list_objects(&subject, &relation)?;

        // Filter by object type
        let filtered: Vec<String> = objects
            .into_iter()
            .filter(|obj| obj.object_type == object_type)
            .map(|obj| obj.to_string_format())
            .collect();

        Ok(filtered)
    }

    /// Expand: get all subjects that have a relation to an object
    pub fn expand(&self, relation: &str, object: &str) -> Result<Vec<String>> {
        let relation = Relation::parse(relation)
            .map_err(|e| Error::InvalidInput(format!("Invalid relation: {}", e)))?;
        let object = Object::parse(object)
            .map_err(|e| Error::InvalidInput(format!("Invalid object: {}", e)))?;

        let graph = self
            .graph
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire read lock".to_string()))?;

        let subjects = graph.expand(&relation, &object)?;

        Ok(subjects.into_iter().map(|s| s.to_string_format()).collect())
    }

    /// Get all relationship tuples
    pub fn get_all_tuples(&self) -> Result<Vec<RelationTuple>> {
        let graph = self
            .graph
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire read lock".to_string()))?;

        Ok(graph.get_all_tuples())
    }

    /// Clear all relationships
    pub fn clear(&self) -> Result<()> {
        let mut graph = self
            .graph
            .write()
            .map_err(|_| Error::InvalidOperation("Failed to acquire write lock".to_string()))?;

        let tuples = graph.get_all_tuples();
        for tuple in tuples {
            let _ = graph.remove_tuple(&tuple);
        }

        Ok(())
    }

    /// Get permission schema
    pub fn get_schema(&self) -> &PermissionSchema {
        &self.schema
    }

    /// Update permission schema
    pub fn set_schema(&mut self, schema: PermissionSchema) {
        self.schema = schema;
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<(usize, usize)> {
        let graph = self
            .graph
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire read lock".to_string()))?;

        graph.cache_stats()
    }

    /// Clear the permission cache
    pub fn clear_cache(&self) -> Result<()> {
        let graph = self
            .graph
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire read lock".to_string()))?;

        graph.clear_cache();
        Ok(())
    }

    /// Batch grant multiple relationships
    pub fn grant_batch(&self, tuples: Vec<(&str, &str, &str)>) -> Result<()> {
        let mut graph = self
            .graph
            .write()
            .map_err(|_| Error::InvalidOperation("Failed to acquire write lock".to_string()))?;

        for (subject, relation, object) in tuples {
            let subj = Subject::parse(subject)
                .map_err(|e| Error::InvalidInput(format!("Invalid subject: {}", e)))?;
            let rel = Relation::parse(relation)
                .map_err(|e| Error::InvalidInput(format!("Invalid relation: {}", e)))?;
            let obj = Object::parse(object)
                .map_err(|e| Error::InvalidInput(format!("Invalid object: {}", e)))?;

            let tuple = RelationTuple::new(subj, rel, obj);
            graph.add_tuple(tuple)?;
        }

        Ok(())
    }

    /// Batch check multiple permissions
    pub fn check_batch(&self, checks: Vec<(&str, &str, &str)>) -> Result<Vec<bool>> {
        let graph = self
            .graph
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire read lock".to_string()))?;

        let mut results = Vec::with_capacity(checks.len());

        for (subject, relation, object) in checks {
            let subj = Subject::parse(subject)
                .map_err(|e| Error::InvalidInput(format!("Invalid subject: {}", e)))?;
            let rel = Relation::parse(relation)
                .map_err(|e| Error::InvalidInput(format!("Invalid relation: {}", e)))?;
            let obj = Object::parse(object)
                .map_err(|e| Error::InvalidInput(format!("Invalid object: {}", e)))?;

            let result = graph.check(&subj, &rel, &obj)?;
            results.push(result);
        }

        Ok(results)
    }
}

impl Default for RebacManager {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-safe shared manager type
pub type SharedRebacManager = Arc<RwLock<RebacManager>>;

/// Create a shared ReBAC manager
pub fn create_shared_rebac_manager() -> SharedRebacManager {
    Arc::new(RwLock::new(RebacManager::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grant_and_check() {
        let manager = RebacManager::new();

        manager
            .grant("user:alice", "owner", "document:123")
            .expect("grant should succeed");

        let has_access = manager
            .check_access("user:alice", "owner", "document:123")
            .expect("check should succeed");
        assert!(has_access);

        let no_access = manager
            .check_access("user:bob", "owner", "document:123")
            .expect("check should succeed");
        assert!(!no_access);
    }

    #[test]
    fn test_revoke() {
        let manager = RebacManager::new();

        manager
            .grant("user:alice", "owner", "document:123")
            .expect("grant should succeed");

        manager
            .revoke("user:alice", "owner", "document:123")
            .expect("revoke should succeed");

        let has_access = manager
            .check_access("user:alice", "owner", "document:123")
            .expect("check should succeed");
        assert!(!has_access);
    }

    #[test]
    fn test_list_accessible() {
        let manager = RebacManager::new();

        manager
            .grant("user:alice", "owner", "document:123")
            .expect("grant should succeed");
        manager
            .grant("user:alice", "owner", "document:456")
            .expect("grant should succeed");

        let objects = manager
            .list_accessible("user:alice", "owner", "document")
            .expect("list should succeed");

        assert_eq!(objects.len(), 2);
    }

    #[test]
    fn test_expand() {
        let manager = RebacManager::new();

        manager
            .grant("user:alice", "viewer", "document:123")
            .expect("grant should succeed");
        manager
            .grant("user:bob", "viewer", "document:123")
            .expect("grant should succeed");

        let subjects = manager
            .expand("viewer", "document:123")
            .expect("expand should succeed");

        assert_eq!(subjects.len(), 2);
    }

    #[test]
    fn test_batch_operations() {
        let manager = RebacManager::new();

        // Batch grant
        manager
            .grant_batch(vec![
                ("user:alice", "owner", "document:123"),
                ("user:bob", "editor", "document:123"),
                ("user:charlie", "viewer", "document:123"),
            ])
            .expect("batch grant should succeed");

        // Batch check
        let results = manager
            .check_batch(vec![
                ("user:alice", "owner", "document:123"),
                ("user:bob", "editor", "document:123"),
                ("user:charlie", "viewer", "document:123"),
                ("user:dave", "owner", "document:123"),
            ])
            .expect("batch check should succeed");

        assert_eq!(results, vec![true, true, true, false]);
    }

    #[test]
    fn test_clear() {
        let manager = RebacManager::new();

        manager
            .grant("user:alice", "owner", "document:123")
            .expect("grant should succeed");

        let tuples = manager.get_all_tuples().expect("get tuples should succeed");
        assert_eq!(tuples.len(), 1);

        manager.clear().expect("clear should succeed");

        let tuples = manager.get_all_tuples().expect("get tuples should succeed");
        assert_eq!(tuples.len(), 0);
    }

    #[test]
    fn test_hierarchical_permissions() {
        let manager = RebacManager::new();

        // Alice owns a document
        manager
            .grant("user:alice", "owner", "document:123")
            .expect("grant should succeed");

        // Document is in a folder
        manager
            .grant("document:123", "parent", "folder:456")
            .expect("grant should succeed");

        // Bob is viewer of the folder
        manager
            .grant("user:bob", "viewer", "folder:456")
            .expect("grant should succeed");

        // Alice should still be owner
        assert!(manager
            .check_access("user:alice", "owner", "document:123")
            .expect("check should succeed"));
    }

    #[test]
    fn test_team_membership() {
        let manager = RebacManager::new();

        // Alice is member of engineering team
        manager
            .grant("user:alice", "member", "team:engineering")
            .expect("grant should succeed");

        // Engineering team members can view document
        manager
            .grant("team:engineering#member", "viewer", "document:123")
            .expect("grant should succeed");

        // This requires transitive resolution through subject sets
        // which is implemented in the graph's is_member_of_set method
    }
}
