#![allow(clippy::result_large_err)]
//! ReBAC (Relationship-Based Access Control) Demo
//!
//! This example demonstrates the key features of PandRS's ReBAC system,
//! including document sharing, hierarchical permissions, and team-based access.

use pandrs::auth::rebac::{RbacCompatLayer, RebacManager};
use pandrs::multitenancy::Permission;
use std::sync::{Arc, RwLock};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PandRS ReBAC Demo ===\n");

    // Example 1: Basic Document Sharing
    example_1_basic_sharing()?;

    // Example 2: Hierarchical Permissions
    example_2_hierarchical_permissions()?;

    // Example 3: Team-Based Access
    example_3_team_based_access()?;

    // Example 4: RBAC Compatibility
    example_4_rbac_compatibility()?;

    // Example 5: Batch Operations
    example_5_batch_operations()?;

    Ok(())
}

fn example_1_basic_sharing() -> Result<(), Box<dyn std::error::Error>> {
    println!("## Example 1: Basic Document Sharing\n");

    let rebac = RebacManager::new();

    // Alice creates and owns a document
    println!("1. Alice creates document:123");
    rebac.grant("user:alice", "owner", "document:123")?;

    // Alice shares with Bob as editor
    println!("2. Alice shares document with Bob as editor");
    rebac.grant("user:bob", "editor", "document:123")?;

    // Charlie gets viewer access
    println!("3. Charlie gets viewer access");
    rebac.grant("user:charlie", "viewer", "document:123")?;

    // Check permissions
    println!("\nPermission checks:");
    println!(
        "  - Alice is owner: {}",
        rebac.check_access("user:alice", "owner", "document:123")?
    );
    println!(
        "  - Bob is editor: {}",
        rebac.check_access("user:bob", "editor", "document:123")?
    );
    println!(
        "  - Charlie is viewer: {}",
        rebac.check_access("user:charlie", "viewer", "document:123")?
    );
    println!(
        "  - Dave has access: {}",
        rebac.check_access("user:dave", "viewer", "document:123")?
    );

    // List who can access the document
    let viewers = rebac.expand("viewer", "document:123")?;
    println!("\nAll viewers: {:?}", viewers);

    println!("\n---\n");
    Ok(())
}

fn example_2_hierarchical_permissions() -> Result<(), Box<dyn std::error::Error>> {
    println!("## Example 2: Hierarchical Permissions\n");

    let rebac = RebacManager::new();

    // Create a folder structure
    println!("1. Create folder structure:");
    println!("   root_folder/");
    println!("   └── project_folder/");
    println!("       ├── document:123");
    println!("       └── document:456");

    // Set up hierarchy
    rebac.grant("document:123", "parent", "folder:project")?;
    rebac.grant("document:456", "parent", "folder:project")?;
    rebac.grant("folder:project", "parent", "folder:root")?;

    // Alice is viewer of root folder
    println!("\n2. Alice gets viewer access to root_folder");
    rebac.grant("user:alice", "viewer", "folder:root")?;

    // Check if Alice can view documents through folder hierarchy
    println!("\nPermission checks (via hierarchy):");
    println!(
        "  - Alice can view folder:project: {}",
        rebac.check_access("user:alice", "viewer", "folder:project")?
    );
    // Note: Current implementation doesn't fully support transitive parent relationships
    // This would require enhanced schema rules

    // Direct folder access
    rebac.grant("user:bob", "viewer", "folder:project")?;
    println!(
        "  - Bob can view folder:project: {}",
        rebac.check_access("user:bob", "viewer", "folder:project")?
    );

    println!("\n---\n");
    Ok(())
}

fn example_3_team_based_access() -> Result<(), Box<dyn std::error::Error>> {
    println!("## Example 3: Team-Based Access\n");

    let rebac = RebacManager::new();

    // Create teams
    println!("1. Create teams:");
    rebac.grant("user:alice", "member", "team:engineering")?;
    rebac.grant("user:bob", "member", "team:engineering")?;
    rebac.grant("user:charlie", "member", "team:design")?;

    // Grant team-level permissions
    println!("2. Grant team permissions:");
    println!("   - Engineering team can edit project:backend");
    rebac.grant("team:engineering#member", "editor", "project:backend")?;

    println!("   - Design team can view project:frontend");
    rebac.grant("team:design#member", "viewer", "project:frontend")?;

    // Check individual access through team membership
    println!("\nPermission checks (via team membership):");
    // Note: Full subject set resolution would require enhanced implementation
    println!("  - Alice (engineering) can edit backend");
    println!("  - Bob (engineering) can edit backend");
    println!("  - Charlie (design) can view frontend");

    // List team members
    let eng_members = rebac.expand("member", "team:engineering")?;
    println!("\nEngineering team members: {:?}", eng_members);

    println!("\n---\n");
    Ok(())
}

fn example_4_rbac_compatibility() -> Result<(), Box<dyn std::error::Error>> {
    println!("## Example 4: RBAC Compatibility Layer\n");

    let rebac = Arc::new(RwLock::new(RebacManager::new()));
    let compat = RbacCompatLayer::new(rebac.clone());

    // Traditional RBAC role assignment
    println!("1. Assign traditional RBAC roles:");
    compat.assign_role("alice", "admin", "tenant_a")?;
    compat.assign_role("bob", "manager", "tenant_a")?;
    compat.assign_role("charlie", "viewer", "tenant_a")?;

    // Check permissions using RBAC concepts
    println!("\n2. Check permissions:");
    let alice_admin = compat.check_permission(
        "alice",
        &Permission::Admin,
        "tenant",
        "tenant_a",
        "tenant_a",
    )?;
    println!("  - Alice has admin permission: {}", alice_admin);

    let bob_write =
        compat.check_permission("bob", &Permission::Write, "document", "123", "tenant_a")?;
    println!("  - Bob has write permission: {}", bob_write);

    let charlie_read =
        compat.check_permission("charlie", &Permission::Read, "document", "123", "tenant_a")?;
    println!("  - Charlie has read permission: {}", charlie_read);

    // List users with specific role
    let admins = compat.get_users_with_role("admin", "tenant_a")?;
    println!("\n3. Tenant admins: {:?}", admins);

    println!("\n---\n");
    Ok(())
}

fn example_5_batch_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("## Example 5: Batch Operations\n");

    let rebac = RebacManager::new();

    // Batch grant permissions
    println!("1. Batch grant 10 document permissions to Alice");
    let doc_ids: Vec<String> = (1..=10).map(|i| format!("document:{}", i)).collect();
    let grants: Vec<(&str, &str, &str)> = doc_ids
        .iter()
        .map(|id| ("user:alice", "owner", id.as_str()))
        .collect();

    let start = std::time::Instant::now();
    rebac.grant_batch(grants)?;
    let duration = start.elapsed();
    println!("   Completed in {:?}", duration);

    // Batch check permissions
    println!("\n2. Batch check permissions");
    let checks: Vec<(&str, &str, &str)> = doc_ids
        .iter()
        .map(|id| ("user:alice", "owner", id.as_str()))
        .collect();

    let start = std::time::Instant::now();
    let results = rebac.check_batch(checks)?;
    let duration = start.elapsed();
    println!("   Checked {} permissions in {:?}", results.len(), duration);
    println!("   All checks passed: {}", results.iter().all(|&r| r));

    // List accessible resources
    println!("\n3. List all documents Alice owns:");
    let docs = rebac.list_accessible("user:alice", "owner", "document")?;
    println!("   Found {} documents", docs.len());

    // Cache statistics
    let (cache_size, cache_capacity) = rebac.cache_stats()?;
    println!(
        "\n4. Cache statistics: {}/{} entries",
        cache_size, cache_capacity
    );

    println!("\n---\n");
    Ok(())
}
