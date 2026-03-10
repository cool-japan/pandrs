//! ReBAC (Relationship-Based Access Control) Module
//!
//! This module implements a modern, flexible authorization system inspired by
//! Google Zanzibar, Auth0 FGA, and Ory Keto. It provides fine-grained access
//! control based on relationships between subjects and objects.
//!
//! # Overview
//!
//! ReBAC extends traditional RBAC by modeling permissions as relationships in a graph.
//! This enables:
//! - Dynamic resource ownership
//! - Transitive permission propagation
//! - Hierarchical resources (folders, organizations)
//! - Flexible sharing and delegation
//! - Subject sets (groups, teams)
//!
//! # Core Concepts
//!
//! ## Relationship Tuples
//!
//! The fundamental unit is a tuple: `(subject, relation, object)`
//!
//! Examples:
//! - `("user:alice", "owner", "document:123")` - Alice owns document 123
//! - `("user:bob", "editor", "document:123")` - Bob can edit document 123
//! - `("team:eng#member", "viewer", "folder:456")` - All engineering team members can view folder 456
//!
//! ## Subjects
//!
//! Entities that can have permissions:
//! - Direct subjects: `"user:alice"`, `"service:api"`
//! - Subject sets: `"team:engineering#member"`, `"group:admins#admin"`
//!
//! ## Relations
//!
//! Types of relationships:
//! - `"owner"` - Full control over a resource
//! - `"editor"` - Can modify a resource
//! - `"viewer"` - Can read a resource
//! - `"member"` - Member of a group/team
//! - `"parent"` - Parent relationship for hierarchies
//!
//! ## Objects
//!
//! Resources that can be accessed:
//! - `"document:123"`, `"folder:456"`, `"tenant:acme"`
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │         ReBAC Manager API               │
//! │  grant() check_access() revoke()        │
//! └─────────────┬───────────────────────────┘
//!               │
//!               ├──► PermissionSchema
//!               │    (defines relation rules)
//!               │
//!               └──► RelationshipGraph
//!                    ├─► Forward Index (object → subjects)
//!                    ├─► Reverse Index (subject → objects)
//!                    └─► LRU Cache (computed permissions)
//! ```
//!
//! # Usage Examples
//!
//! ## Basic Usage
//!
//! ```ignore
//! use pandrs::auth::rebac::{RebacManager};
//!
//! let rebac = RebacManager::new();
//!
//! // Grant permission
//! rebac.grant("user:alice", "owner", "document:123")?;
//!
//! // Check permission
//! let can_access = rebac.check_access("user:alice", "owner", "document:123")?;
//! assert!(can_access);
//!
//! // Revoke permission
//! rebac.revoke("user:alice", "owner", "document:123")?;
//! ```
//!
//! ## Document Sharing
//!
//! ```ignore
//! let rebac = RebacManager::new();
//!
//! // Alice creates and owns a document
//! rebac.grant("user:alice", "owner", "document:123")?;
//!
//! // Alice shares with Bob as editor
//! rebac.grant("user:bob", "editor", "document:123")?;
//!
//! // Check access
//! assert!(rebac.check_access("user:bob", "editor", "document:123")?);
//! ```
//!
//! ## Hierarchical Permissions
//!
//! ```ignore
//! let rebac = RebacManager::new();
//!
//! // Create folder hierarchy
//! rebac.grant("document:123", "parent", "folder:456")?;
//! rebac.grant("user:alice", "viewer", "folder:456")?;
//!
//! // Alice can view document through folder permission
//! assert!(rebac.check_access("user:alice", "viewer", "document:123")?);
//! ```
//!
//! ## Team-Based Access
//!
//! ```ignore
//! let rebac = RebacManager::new();
//!
//! // Alice is member of engineering team
//! rebac.grant("user:alice", "member", "team:engineering")?;
//!
//! // Team members can view project
//! rebac.grant("team:engineering#member", "viewer", "project:secret")?;
//!
//! // Alice can view through team membership
//! assert!(rebac.check_access("user:alice", "viewer", "project:secret")?);
//! ```
//!
//! ## RBAC Compatibility
//!
//! ```ignore
//! use pandrs::auth::rebac::{RebacManager, RbacCompatLayer};
//! use pandrs::multitenancy::Permission;
//! use std::sync::{Arc, RwLock};
//!
//! let rebac = Arc::new(RwLock::new(RebacManager::new()));
//! let compat = RbacCompatLayer::new(rebac);
//!
//! // Assign traditional RBAC role
//! compat.assign_role("alice", "admin", "tenant_a")?;
//!
//! // Check permission using RBAC concepts
//! let can_admin = compat.check_permission(
//!     "alice",
//!     &Permission::Admin,
//!     "tenant",
//!     "tenant_a",
//!     "tenant_a"
//! )?;
//! assert!(can_admin);
//! ```
//!
//! # Performance
//!
//! - **Direct checks**: O(1) with cache, O(log n) without
//! - **Transitive checks**: O(d × b) where d=depth, b=branching factor
//! - **Caching**: LRU cache for computed permissions (default 1000 entries)
//! - **Typical latency**: < 10ms for most queries with warm cache
//!
//! # Thread Safety
//!
//! All components are thread-safe using `RwLock` for concurrent access.
//! Multiple readers can check permissions simultaneously, while writes
//! (grant/revoke) acquire exclusive locks.
//!
//! # Future Enhancements
//!
//! - Persistent storage backends (PostgreSQL, RocksDB)
//! - Audit logging for relationship changes
//! - Bulk operations optimization
//! - Watch API for relationship changes
//! - gRPC API compatibility with Zanzibar

pub mod compat;
pub mod graph;
pub mod manager;
pub mod schema;
pub mod types;

// Re-export main types for convenience
pub use compat::RbacCompatLayer;
pub use graph::RelationshipGraph;
pub use manager::{create_shared_rebac_manager, RebacManager, SharedRebacManager};
pub use schema::{PermissionSchema, RelationDefinition, RewriteRule, TypeDefinition};
pub use types::{Object, Relation, RelationTuple, Subject};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_document_sharing() {
        let rebac = RebacManager::new();

        // Alice creates a document
        rebac
            .grant("user:alice", "owner", "document:123")
            .expect("grant should succeed");

        // Alice shares with Bob as editor
        rebac
            .grant("user:bob", "editor", "document:123")
            .expect("grant should succeed");

        // Charlie gets viewer access
        rebac
            .grant("user:charlie", "viewer", "document:123")
            .expect("grant should succeed");

        // Verify permissions
        assert!(rebac
            .check_access("user:alice", "owner", "document:123")
            .expect("check should succeed"));
        assert!(rebac
            .check_access("user:bob", "editor", "document:123")
            .expect("check should succeed"));
        assert!(rebac
            .check_access("user:charlie", "viewer", "document:123")
            .expect("check should succeed"));

        // Dave has no access
        assert!(!rebac
            .check_access("user:dave", "viewer", "document:123")
            .expect("check should succeed"));
    }

    #[test]
    fn test_folder_hierarchy() {
        let rebac = RebacManager::new();

        // Create hierarchy: document -> folder -> parent_folder
        rebac
            .grant("document:123", "parent", "folder:456")
            .expect("grant should succeed");
        rebac
            .grant("folder:456", "parent", "folder:789")
            .expect("grant should succeed");

        // Alice has viewer permission on parent folder
        rebac
            .grant("user:alice", "viewer", "folder:789")
            .expect("grant should succeed");

        // Alice should be able to view through hierarchy
        assert!(rebac
            .check_access("user:alice", "viewer", "folder:456")
            .expect("check should succeed"));
    }

    #[test]
    fn test_batch_operations_performance() {
        let rebac = RebacManager::new();

        // Create document IDs first to own the strings
        let doc_ids: Vec<String> = (0..100).map(|i| format!("document:{}", i)).collect();

        // Batch grant 100 permissions
        let grants: Vec<(&str, &str, &str)> = doc_ids
            .iter()
            .map(|doc_id| ("user:alice", "viewer", doc_id.as_str()))
            .collect();

        let start = std::time::Instant::now();
        rebac
            .grant_batch(grants)
            .expect("batch grant should succeed");
        let grant_duration = start.elapsed();

        println!("Batch grant (100 items): {:?}", grant_duration);

        // Batch check 100 permissions
        let checks: Vec<(&str, &str, &str)> = doc_ids
            .iter()
            .map(|doc_id| ("user:alice", "viewer", doc_id.as_str()))
            .collect();

        let start = std::time::Instant::now();
        let results = rebac
            .check_batch(checks)
            .expect("batch check should succeed");
        let check_duration = start.elapsed();

        println!("Batch check (100 items): {:?}", check_duration);

        assert_eq!(results.len(), 100);
        assert!(results.iter().all(|&r| r));
    }

    #[test]
    fn test_list_accessible_resources() {
        let rebac = RebacManager::new();

        // Grant Alice access to multiple documents
        for i in 1..=5 {
            rebac
                .grant("user:alice", "owner", &format!("document:{}", i))
                .expect("grant should succeed");
        }

        // List all documents Alice owns
        let docs = rebac
            .list_accessible("user:alice", "owner", "document")
            .expect("list should succeed");

        assert_eq!(docs.len(), 5);
    }

    #[test]
    fn test_expand_permissions() {
        let rebac = RebacManager::new();

        // Multiple users have access to the same document
        rebac
            .grant("user:alice", "viewer", "document:123")
            .expect("grant should succeed");
        rebac
            .grant("user:bob", "viewer", "document:123")
            .expect("grant should succeed");
        rebac
            .grant("user:charlie", "viewer", "document:123")
            .expect("grant should succeed");

        // Expand to see all viewers
        let viewers = rebac
            .expand("viewer", "document:123")
            .expect("expand should succeed");

        assert_eq!(viewers.len(), 3);
    }

    #[test]
    fn test_cache_effectiveness() {
        let rebac = RebacManager::with_cache_size(100);

        rebac
            .grant("user:alice", "owner", "document:123")
            .expect("grant should succeed");

        // First check (cache miss)
        let start = std::time::Instant::now();
        let result1 = rebac
            .check_access("user:alice", "owner", "document:123")
            .expect("check should succeed");
        let duration1 = start.elapsed();

        // Second check (cache hit - should be faster)
        let start = std::time::Instant::now();
        let result2 = rebac
            .check_access("user:alice", "owner", "document:123")
            .expect("check should succeed");
        let duration2 = start.elapsed();

        assert!(result1);
        assert!(result2);

        println!("First check (miss): {:?}", duration1);
        println!("Second check (hit): {:?}", duration2);

        // Verify cache stats
        let (cache_size, cache_capacity) = rebac.cache_stats().expect("stats should succeed");
        assert_eq!(cache_size, 1);
        assert_eq!(cache_capacity, 100);
    }
}
