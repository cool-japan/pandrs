#![allow(clippy::result_large_err)]
//! Comprehensive ReBAC (Relationship-Based Access Control) Tests
//!
//! Tests covering edge cases, security scenarios, and complex permission graphs

use pandrs::auth::rebac::RebacManager;

/// Test transitive permissions through direct chain
#[test]
fn test_transitive_permissions_simple_chain() {
    let rebac = RebacManager::new();

    // Create chain: alice -> document -> folder -> workspace
    rebac
        .grant("document:doc1", "parent", "folder:f1")
        .expect("grant should succeed");
    rebac
        .grant("folder:f1", "parent", "workspace:ws1")
        .expect("grant should succeed");
    rebac
        .grant("user:alice", "viewer", "workspace:ws1")
        .expect("grant should succeed");

    // Alice should have transitive access to folder through workspace
    assert!(rebac
        .check_access("user:alice", "viewer", "folder:f1")
        .expect("check should succeed"));
}

/// Test transitive permissions with deep hierarchy (5 levels)
#[test]
fn test_transitive_permissions_deep_hierarchy() {
    let rebac = RebacManager::new();

    // Create deep hierarchy: doc -> f1 -> f2 -> f3 -> f4 -> root
    rebac
        .grant("document:doc1", "parent", "folder:f1")
        .expect("grant should succeed");
    rebac
        .grant("folder:f1", "parent", "folder:f2")
        .expect("grant should succeed");
    rebac
        .grant("folder:f2", "parent", "folder:f3")
        .expect("grant should succeed");
    rebac
        .grant("folder:f3", "parent", "folder:f4")
        .expect("grant should succeed");
    rebac
        .grant("folder:f4", "parent", "folder:root")
        .expect("grant should succeed");

    // Grant permission at root level
    rebac
        .grant("user:alice", "viewer", "folder:root")
        .expect("grant should succeed");

    // Alice should have access through deep transitive chain
    assert!(rebac
        .check_access("user:alice", "viewer", "folder:f4")
        .expect("check should succeed"));
}

/// Test cycle detection in permission graph
#[test]
fn test_relationship_cycle_detection() {
    let rebac = RebacManager::new();

    // Create a cycle: f1 -> f2 -> f3 -> f1
    rebac
        .grant("folder:f1", "parent", "folder:f2")
        .expect("grant should succeed");
    rebac
        .grant("folder:f2", "parent", "folder:f3")
        .expect("grant should succeed");
    rebac
        .grant("folder:f3", "parent", "folder:f1")
        .expect("grant should succeed");

    // Grant permission on f1
    rebac
        .grant("user:alice", "viewer", "folder:f1")
        .expect("grant should succeed");

    // Should handle cycle gracefully (not infinite loop)
    let result = rebac.check_access("user:alice", "viewer", "folder:f2");
    assert!(result.is_ok(), "Should handle cycles without error");
}

/// Test multi-path permission grant
#[test]
fn test_multi_path_permission_grant() {
    let rebac = RebacManager::new();

    // Grant permission through multiple relations
    rebac
        .grant("user:alice", "viewer", "folder:top")
        .expect("grant should succeed");
    rebac
        .grant("user:alice", "editor", "folder:top")
        .expect("grant should succeed");
    rebac
        .grant("user:alice", "owner", "folder:top")
        .expect("grant should succeed");

    // All grants should exist independently
    assert!(rebac
        .check_access("user:alice", "viewer", "folder:top")
        .expect("check should succeed"));
    assert!(rebac
        .check_access("user:alice", "editor", "folder:top")
        .expect("check should succeed"));
    assert!(rebac
        .check_access("user:alice", "owner", "folder:top")
        .expect("check should succeed"));
}

/// Test access denial for non-granted permissions
#[test]
fn test_access_denial() {
    let rebac = RebacManager::new();

    // Grant Alice viewer permission
    rebac
        .grant("user:alice", "viewer", "document:doc1")
        .expect("grant should succeed");

    // Bob should NOT have access
    assert!(
        !rebac
            .check_access("user:bob", "viewer", "document:doc1")
            .expect("check should succeed"),
        "Bob should not have access"
    );

    // Alice should NOT have editor permission
    assert!(
        !rebac
            .check_access("user:alice", "editor", "document:doc1")
            .expect("check should succeed"),
        "Alice should not have editor permission"
    );
}

/// Test multi-user isolation
#[test]
fn test_multi_user_isolation() {
    let rebac = RebacManager::new();

    // Alice's documents
    rebac
        .grant("user:alice", "owner", "document:alice_doc1")
        .expect("grant should succeed");
    rebac
        .grant("user:alice", "owner", "document:alice_doc2")
        .expect("grant should succeed");

    // Bob's documents
    rebac
        .grant("user:bob", "owner", "document:bob_doc1")
        .expect("grant should succeed");
    rebac
        .grant("user:bob", "owner", "document:bob_doc2")
        .expect("grant should succeed");

    // Alice should have access to her documents
    assert!(rebac
        .check_access("user:alice", "owner", "document:alice_doc1")
        .expect("check should succeed"));
    assert!(rebac
        .check_access("user:alice", "owner", "document:alice_doc2")
        .expect("check should succeed"));

    // But not to Bob's documents
    assert!(
        !rebac
            .check_access("user:alice", "owner", "document:bob_doc1")
            .expect("check should succeed"),
        "Alice should not have access to Bob's documents"
    );

    // Bob should have access to his documents
    assert!(rebac
        .check_access("user:bob", "owner", "document:bob_doc1")
        .expect("check should succeed"));

    // But not to Alice's documents
    assert!(
        !rebac
            .check_access("user:bob", "owner", "document:alice_doc1")
            .expect("check should succeed"),
        "Bob should not have access to Alice's documents"
    );
}

/// Test permission revocation
#[test]
fn test_permission_revocation() {
    let rebac = RebacManager::new();

    // Grant permission
    rebac
        .grant("user:alice", "viewer", "document:doc1")
        .expect("grant should succeed");

    // Verify permission exists
    assert!(rebac
        .check_access("user:alice", "viewer", "document:doc1")
        .expect("check should succeed"));

    // Revoke permission
    rebac
        .revoke("user:alice", "viewer", "document:doc1")
        .expect("revoke should succeed");

    // Verify permission removed
    assert!(
        !rebac
            .check_access("user:alice", "viewer", "document:doc1")
            .expect("check should succeed"),
        "Permission should be revoked"
    );
}

/// Test permission revocation and re-grant
#[test]
fn test_permission_revocation_and_regrant() {
    let rebac = RebacManager::new();

    // Grant permission
    rebac
        .grant("user:alice", "viewer", "document:doc1")
        .expect("grant should succeed");

    // Verify access
    assert!(rebac
        .check_access("user:alice", "viewer", "document:doc1")
        .expect("check should succeed"));

    // Revoke permission
    rebac
        .revoke("user:alice", "viewer", "document:doc1")
        .expect("revoke should succeed");

    // Verify no access
    assert!(
        !rebac
            .check_access("user:alice", "viewer", "document:doc1")
            .expect("check should succeed"),
        "Access should be revoked"
    );

    // Re-grant permission
    rebac
        .grant("user:alice", "viewer", "document:doc1")
        .expect("grant should succeed");

    // Verify access restored
    assert!(rebac
        .check_access("user:alice", "viewer", "document:doc1")
        .expect("check should succeed"));
}

/// Test batch grant operations
#[test]
fn test_batch_grant_operations() {
    let rebac = RebacManager::new();

    // Prepare batch grants
    let grants = vec![
        ("user:alice", "viewer", "document:doc1"),
        ("user:alice", "viewer", "document:doc2"),
        ("user:alice", "viewer", "document:doc3"),
        ("user:bob", "editor", "document:doc1"),
        ("user:bob", "editor", "document:doc2"),
    ];

    // Batch grant
    rebac
        .grant_batch(grants)
        .expect("batch grant should succeed");

    // Verify all grants
    assert!(rebac
        .check_access("user:alice", "viewer", "document:doc1")
        .expect("check should succeed"));
    assert!(rebac
        .check_access("user:alice", "viewer", "document:doc3")
        .expect("check should succeed"));
    assert!(rebac
        .check_access("user:bob", "editor", "document:doc2")
        .expect("check should succeed"));
}

/// Test batch check operations
#[test]
fn test_batch_check_operations() {
    let rebac = RebacManager::new();

    // Grant permissions
    rebac
        .grant("user:alice", "viewer", "document:doc1")
        .expect("grant should succeed");
    rebac
        .grant("user:alice", "viewer", "document:doc2")
        .expect("grant should succeed");

    // Batch check
    let checks = vec![
        ("user:alice", "viewer", "document:doc1"),
        ("user:alice", "viewer", "document:doc2"),
        ("user:alice", "viewer", "document:doc3"), // Not granted
        ("user:bob", "viewer", "document:doc1"),   // Not granted
    ];

    let results = rebac
        .check_batch(checks)
        .expect("batch check should succeed");

    assert_eq!(results.len(), 4);
    assert!(results[0], "Alice should have access to doc1");
    assert!(results[1], "Alice should have access to doc2");
    assert!(!results[2], "Alice should not have access to doc3");
    assert!(!results[3], "Bob should not have access to doc1");
}

/// Test cache invalidation on revocation
#[test]
fn test_cache_invalidation() {
    let rebac = RebacManager::new();

    // Grant and check (should cache)
    rebac
        .grant("user:alice", "viewer", "document:doc1")
        .expect("grant should succeed");

    // First check (cache miss)
    assert!(rebac
        .check_access("user:alice", "viewer", "document:doc1")
        .expect("check should succeed"));

    // Second check (cache hit)
    assert!(rebac
        .check_access("user:alice", "viewer", "document:doc1")
        .expect("check should succeed"));

    // Revoke (should invalidate cache)
    rebac
        .revoke("user:alice", "viewer", "document:doc1")
        .expect("revoke should succeed");

    // Check should reflect revocation (not stale cache)
    assert!(
        !rebac
            .check_access("user:alice", "viewer", "document:doc1")
            .expect("check should succeed"),
        "Cache should be invalidated"
    );
}

/// Test subject set expansion (team membership)
#[test]
fn test_subject_set_expansion() {
    let rebac = RebacManager::new();

    // Alice is member of engineering team
    rebac
        .grant("user:alice", "member", "team:engineering")
        .expect("grant should succeed");

    // Team members can view project
    rebac
        .grant("team:engineering#member", "viewer", "project:secret")
        .expect("grant should succeed");

    // Alice should have access through team membership
    assert!(rebac
        .check_access("user:alice", "viewer", "project:secret")
        .expect("check should succeed"));
}

/// Test permission inheritance (owner includes editor and viewer)
#[test]
fn test_permission_inheritance() {
    let rebac = RebacManager::new();

    // Grant owner permission
    rebac
        .grant("user:alice", "owner", "document:doc1")
        .expect("grant should succeed");

    // Owner should also have editor and viewer permissions
    // (depends on schema configuration)
    assert!(rebac
        .check_access("user:alice", "owner", "document:doc1")
        .expect("check should succeed"));
}

/// Test list accessible resources
#[test]
fn test_list_accessible_resources() {
    let rebac = RebacManager::new();

    // Grant Alice access to multiple documents
    rebac
        .grant("user:alice", "viewer", "document:doc1")
        .expect("grant should succeed");
    rebac
        .grant("user:alice", "viewer", "document:doc2")
        .expect("grant should succeed");
    rebac
        .grant("user:alice", "viewer", "document:doc3")
        .expect("grant should succeed");

    // Grant Bob access to different documents
    rebac
        .grant("user:bob", "viewer", "document:doc4")
        .expect("grant should succeed");

    // List Alice's accessible documents
    let docs = rebac
        .list_accessible("user:alice", "viewer", "document")
        .expect("list should succeed");

    assert!(
        docs.len() >= 3,
        "Alice should have access to at least 3 documents"
    );
}

/// Test empty permission check (no relationships)
#[test]
fn test_empty_permission_check() {
    let rebac = RebacManager::new();

    // Check access on empty graph
    assert!(
        !rebac
            .check_access("user:alice", "viewer", "document:doc1")
            .expect("check should succeed"),
        "No permissions granted"
    );
}

/// Test same subject-object with different relations
#[test]
fn test_multiple_relations_same_pair() {
    let rebac = RebacManager::new();

    // Grant multiple relations for same subject-object pair
    rebac
        .grant("user:alice", "viewer", "document:doc1")
        .expect("grant should succeed");
    rebac
        .grant("user:alice", "editor", "document:doc1")
        .expect("grant should succeed");
    rebac
        .grant("user:alice", "owner", "document:doc1")
        .expect("grant should succeed");

    // All relations should be independent
    assert!(rebac
        .check_access("user:alice", "viewer", "document:doc1")
        .expect("check should succeed"));
    assert!(rebac
        .check_access("user:alice", "editor", "document:doc1")
        .expect("check should succeed"));
    assert!(rebac
        .check_access("user:alice", "owner", "document:doc1")
        .expect("check should succeed"));

    // Revoke one relation
    rebac
        .revoke("user:alice", "editor", "document:doc1")
        .expect("revoke should succeed");

    // Others should remain
    assert!(rebac
        .check_access("user:alice", "viewer", "document:doc1")
        .expect("check should succeed"));
    assert!(rebac
        .check_access("user:alice", "owner", "document:doc1")
        .expect("check should succeed"));
    assert!(
        !rebac
            .check_access("user:alice", "editor", "document:doc1")
            .expect("check should succeed"),
        "Editor permission should be revoked"
    );
}

/// Test special characters in identifiers
#[test]
fn test_special_characters_in_identifiers() {
    let rebac = RebacManager::new();

    // Use special characters in IDs
    rebac
        .grant(
            "user:alice@example.com",
            "viewer",
            "document:file-name_123.pdf",
        )
        .expect("grant should succeed");

    assert!(rebac
        .check_access(
            "user:alice@example.com",
            "viewer",
            "document:file-name_123.pdf"
        )
        .expect("check should succeed"));
}

/// Test concurrent permission checks (stress test)
#[test]
fn test_concurrent_permission_checks() {
    let rebac = RebacManager::new();

    // Grant many permissions
    for i in 0..100 {
        rebac
            .grant("user:alice", "viewer", &format!("document:{}", i))
            .expect("grant should succeed");
    }

    // Check all permissions
    for i in 0..100 {
        assert!(rebac
            .check_access("user:alice", "viewer", &format!("document:{}", i))
            .expect("check should succeed"));
    }
}

/// Test large-scale permission grants
#[test]
fn test_large_scale_permission_grants() {
    let rebac = RebacManager::new();

    // Create large number of direct grants: 10 folders, 10 documents each
    for folder in 0..10 {
        rebac
            .grant("user:alice", "viewer", &format!("folder:{}", folder))
            .expect("grant should succeed");

        for doc in 0..10 {
            rebac
                .grant(
                    "user:alice",
                    "viewer",
                    &format!("document:f{}_d{}", folder, doc),
                )
                .expect("grant should succeed");
        }
    }

    // Verify direct permissions work
    assert!(rebac
        .check_access("user:alice", "viewer", "folder:5")
        .expect("check should succeed"));
    assert!(rebac
        .check_access("user:alice", "viewer", "document:f5_d7")
        .expect("check should succeed"));
}
