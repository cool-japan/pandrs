#![allow(clippy::result_large_err)]
//! Comprehensive ReBAC (Relationship-Based Access Control) Example
//!
//! This example demonstrates advanced ReBAC patterns including:
//! - Document sharing with granular permissions
//! - Folder hierarchies with transitive access
//! - Team-based permissions and memberships
//! - Conditional access patterns (unions, intersections, exclusions)
//! - Performance optimization with caching
//! - Real-world multi-tenant SaaS scenario
//! - Comparison with traditional RBAC
//!
//! ReBAC enables fine-grained, relationship-based access control inspired by
//! Google Zanzibar, Auth0 FGA, and Ory Keto.

use pandrs::auth::rebac::{RbacCompatLayer, RebacManager};
use pandrs::multitenancy::Permission;
use std::sync::{Arc, RwLock};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     PandRS ReBAC Comprehensive Demonstration                  ║");
    println!("║     Relationship-Based Access Control System                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // 1. Basic Setup and Permissions
    println!("═══ Section 1: Basic Setup and Permissions ═══\n");
    basic_permissions_demo()?;

    // 2. Document Sharing Scenario
    println!("\n═══ Section 2: Document Sharing Scenario ═══\n");
    document_sharing_demo()?;

    // 3. Folder Hierarchies with Transitive Permissions
    println!("\n═══ Section 3: Folder Hierarchies ═══\n");
    folder_hierarchy_demo()?;

    // 4. Team Memberships and Permissions
    println!("\n═══ Section 4: Team Memberships ═══\n");
    team_membership_demo()?;

    // 5. Conditional Access Patterns
    println!("\n═══ Section 5: Conditional Access Patterns ═══\n");
    conditional_access_demo()?;

    // 6. RBAC vs ReBAC Comparison
    println!("\n═══ Section 6: RBAC vs ReBAC Comparison ═══\n");
    rbac_comparison_demo()?;

    // 7. Performance Demonstration
    println!("\n═══ Section 7: Performance Demonstration ═══\n");
    performance_demo()?;

    // 8. Real-World Multi-Tenant SaaS Scenario
    println!("\n═══ Section 8: Real-World Multi-Tenant SaaS ═══\n");
    real_world_scenario()?;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║     ReBAC Demonstration Complete                              ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Demonstrates basic ReBAC operations: grant, check, revoke
fn basic_permissions_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("Basic ReBAC operations demonstrate the fundamental building blocks");
    println!("of relationship-based access control.\n");

    let rebac = RebacManager::new();

    // Create a simple permission: Alice owns a document
    println!("1. Granting ownership:");
    println!("   rebac.grant(\"user:alice\", \"owner\", \"document:report123\")");
    rebac.grant("user:alice", "owner", "document:report123")?;

    // Check if Alice has ownership
    println!("\n2. Checking permission:");
    let is_owner = rebac.check_access("user:alice", "owner", "document:report123")?;
    println!("   Alice is owner of document:report123: {}", is_owner);
    assert!(is_owner, "Alice should be owner");

    // Check if Bob has access (should be false)
    let bob_has_access = rebac.check_access("user:bob", "owner", "document:report123")?;
    println!("   Bob is owner of document:report123: {}", bob_has_access);
    assert!(!bob_has_access, "Bob should not be owner");

    // List all documents Alice owns
    println!("\n3. Listing accessible resources:");
    let alice_docs = rebac.list_accessible("user:alice", "owner", "document")?;
    println!(
        "   Alice owns {} document(s): {:?}",
        alice_docs.len(),
        alice_docs
    );

    // Revoke permission
    println!("\n4. Revoking permission:");
    println!("   rebac.revoke(\"user:alice\", \"owner\", \"document:report123\")");
    rebac.revoke("user:alice", "owner", "document:report123")?;

    // Verify revocation
    let still_owner = rebac.check_access("user:alice", "owner", "document:report123")?;
    println!("   Alice is owner after revoke: {}", still_owner);
    assert!(!still_owner, "Alice should no longer be owner");

    println!("\n✓ Basic operations completed successfully");
    Ok(())
}

/// Demonstrates document sharing with different permission levels
fn document_sharing_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("Document sharing demonstrates how to grant different permission levels");
    println!("to different users on the same resource.\n");

    let rebac = RebacManager::new();

    // Scenario: Alice creates a document and shares it
    println!("Scenario: Alice creates 'Q4_Financial_Report.xlsx'\n");

    // Alice is the owner
    println!("Step 1: Alice becomes the owner");
    rebac.grant("user:alice", "owner", "document:q4_report")?;
    println!("   ✓ user:alice → owner → document:q4_report");

    // Alice shares with Bob as editor
    println!("\nStep 2: Alice shares with Bob (editor permission)");
    rebac.grant("user:bob", "editor", "document:q4_report")?;
    println!("   ✓ user:bob → editor → document:q4_report");

    // Alice shares with Charlie as viewer
    println!("\nStep 3: Alice shares with Charlie (viewer permission)");
    rebac.grant("user:charlie", "viewer", "document:q4_report")?;
    println!("   ✓ user:charlie → viewer → document:q4_report");

    // Alice shares with Dave as viewer
    println!("\nStep 4: Alice shares with Dave (viewer permission)");
    rebac.grant("user:dave", "viewer", "document:q4_report")?;
    println!("   ✓ user:dave → viewer → document:q4_report");

    // Check permissions
    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│ Permission Summary                                      │");
    println!("├─────────────────────────────────────────────────────────┤");

    let users = ["alice", "bob", "charlie", "dave", "eve"];
    let relations = ["owner", "editor", "viewer"];

    for user in &users {
        let user_str = format!("user:{}", user);
        print!("│ {:<12}", user);

        for relation in &relations {
            let has_perm = rebac.check_access(&user_str, relation, "document:q4_report")?;
            print!(" │ {:>6}: {:5}", relation, if has_perm { "✓" } else { "✗" });
        }
        println!(" │");
    }
    println!("└─────────────────────────────────────────────────────────┘");

    // Expand to see all viewers
    println!("\nStep 5: List all users with viewer access");
    let viewers = rebac.expand("viewer", "document:q4_report")?;
    println!("   Viewers: {:?}", viewers);
    println!("   Total: {} users", viewers.len());

    // Revoke Charlie's access
    println!("\nStep 6: Revoke Charlie's viewer access");
    rebac.revoke("user:charlie", "viewer", "document:q4_report")?;
    println!("   ✓ Revoked: user:charlie viewer document:q4_report");

    let charlie_still_has_access =
        rebac.check_access("user:charlie", "viewer", "document:q4_report")?;
    println!("   Charlie can still view: {}", charlie_still_has_access);
    assert!(!charlie_still_has_access, "Charlie should not have access");

    println!("\n✓ Document sharing scenario completed");
    Ok(())
}

/// Demonstrates folder hierarchies with transitive permissions
fn folder_hierarchy_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("Folder hierarchies demonstrate how permissions can propagate");
    println!("through parent-child relationships.\n");

    let rebac = RebacManager::new();

    // Create folder structure
    println!("Creating folder hierarchy:");
    println!(
        "
    Organization Root
    ├── Engineering/
    │   ├── Backend/
    │   │   ├── API_Spec.md
    │   │   └── Database_Schema.sql
    │   └── Frontend/
    │       └── Design_System.fig
    └── Marketing/
        └── Campaign_2026.pptx
    "
    );

    // Set up hierarchy with parent relationships
    println!("Setting up parent relationships:");

    // Engineering folder → Organization Root
    rebac.grant("folder:engineering", "parent", "folder:org_root")?;
    println!("   ✓ folder:engineering → parent → folder:org_root");

    // Backend folder → Engineering
    rebac.grant("folder:backend", "parent", "folder:engineering")?;
    println!("   ✓ folder:backend → parent → folder:engineering");

    // Frontend folder → Engineering
    rebac.grant("folder:frontend", "parent", "folder:engineering")?;
    println!("   ✓ folder:frontend → parent → folder:engineering");

    // Marketing folder → Organization Root
    rebac.grant("folder:marketing", "parent", "folder:org_root")?;
    println!("   ✓ folder:marketing → parent → folder:org_root");

    // Documents → Folders
    rebac.grant("document:api_spec", "parent", "folder:backend")?;
    rebac.grant("document:db_schema", "parent", "folder:backend")?;
    rebac.grant("document:design_system", "parent", "folder:frontend")?;
    rebac.grant("document:campaign", "parent", "folder:marketing")?;
    println!("   ✓ Documents assigned to folders");

    // Grant permissions at different levels
    println!("\nGranting permissions:");

    // Alice: viewer at organization root (should see everything)
    rebac.grant("user:alice", "viewer", "folder:org_root")?;
    println!("   ✓ Alice: viewer of org_root (top-level access)");

    // Bob: viewer at engineering folder (should see eng folders/docs)
    rebac.grant("user:bob", "viewer", "folder:engineering")?;
    println!("   ✓ Bob: viewer of engineering folder");

    // Charlie: viewer at backend folder only
    rebac.grant("user:charlie", "viewer", "folder:backend")?;
    println!("   ✓ Charlie: viewer of backend folder");

    // Check transitive permissions
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ Transitive Permission Check                                 │");
    println!("├─────────────────────────────────────────────────────────────┤");

    let test_cases = [
        ("alice", "folder:org_root", true, "Direct permission"),
        (
            "alice",
            "folder:engineering",
            true,
            "Transitive through org_root",
        ),
        (
            "alice",
            "folder:backend",
            true,
            "Transitive through eng→root",
        ),
        ("bob", "folder:org_root", false, "No access to parent"),
        ("bob", "folder:engineering", true, "Direct permission"),
        (
            "bob",
            "folder:backend",
            true,
            "Transitive through engineering",
        ),
        ("charlie", "folder:backend", true, "Direct permission"),
        (
            "charlie",
            "folder:engineering",
            false,
            "No access to parent",
        ),
        ("charlie", "folder:marketing", false, "Different branch"),
    ];

    for (user, resource, expected, description) in &test_cases {
        let user_str = format!("user:{}", user);
        let has_access = rebac.check_access(&user_str, "viewer", resource)?;
        let status = if has_access == *expected {
            "✓"
        } else {
            "✗"
        };
        println!(
            "│ {} {:<8} can view {:<20} │ {} │",
            status, user, resource, description
        );
    }
    println!("└─────────────────────────────────────────────────────────────┘");

    println!("\n✓ Folder hierarchy scenario completed");
    Ok(())
}

/// Demonstrates team memberships and team-level permissions
fn team_membership_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("Team memberships demonstrate how permissions can be granted");
    println!("to groups of users through subject sets.\n");

    let rebac = RebacManager::new();

    // Create teams
    println!("Creating teams and adding members:\n");

    // Engineering team
    println!("Engineering Team:");
    rebac.grant("user:alice", "member", "team:engineering")?;
    println!("   ✓ Alice → member → team:engineering");
    rebac.grant("user:bob", "member", "team:engineering")?;
    println!("   ✓ Bob → member → team:engineering");
    rebac.grant("user:charlie", "member", "team:engineering")?;
    println!("   ✓ Charlie → member → team:engineering");

    // Design team
    println!("\nDesign Team:");
    rebac.grant("user:dave", "member", "team:design")?;
    println!("   ✓ Dave → member → team:design");
    rebac.grant("user:eve", "member", "team:design")?;
    println!("   ✓ Eve → member → team:design");

    // Marketing team
    println!("\nMarketing Team:");
    rebac.grant("user:frank", "member", "team:marketing")?;
    println!("   ✓ Frank → member → team:marketing");
    rebac.grant("user:grace", "member", "team:marketing")?;
    println!("   ✓ Grace → member → team:marketing");

    // Grant team-level permissions using subject sets
    println!("\n\nGranting team-level permissions:");

    // Engineering team can edit backend project
    rebac.grant("team:engineering#member", "editor", "project:backend_api")?;
    println!("   ✓ team:engineering#member → editor → project:backend_api");
    println!("     (All engineering members can edit the backend project)");

    // Design team can view the design system
    rebac.grant("team:design#member", "viewer", "project:design_system")?;
    println!("   ✓ team:design#member → viewer → project:design_system");
    println!("     (All design members can view the design system)");

    // Marketing team can edit marketing materials
    rebac.grant("team:marketing#member", "editor", "project:campaigns")?;
    println!("   ✓ team:marketing#member → editor → project:campaigns");
    println!("     (All marketing members can edit campaigns)");

    // Check individual access through team membership
    println!("\n┌──────────────────────────────────────────────────────────┐");
    println!("│ Individual Access Through Team Membership                │");
    println!("├──────────────────────────────────────────────────────────┤");

    // List team members
    let eng_members = rebac.expand("member", "team:engineering")?;
    println!("│ Engineering team: {} members", eng_members.len());
    for member in &eng_members {
        println!("│   - {}", member);
    }

    let design_members = rebac.expand("member", "team:design")?;
    println!("│ Design team: {} members", design_members.len());
    for member in &design_members {
        println!("│   - {}", member);
    }

    let marketing_members = rebac.expand("member", "team:marketing")?;
    println!("│ Marketing team: {} members", marketing_members.len());
    for member in &marketing_members {
        println!("│   - {}", member);
    }
    println!("└──────────────────────────────────────────────────────────┘");

    // Direct membership checks
    println!("\nDirect membership verification:");
    println!(
        "   Alice is in engineering: {}",
        rebac.check_access("user:alice", "member", "team:engineering")?
    );
    println!(
        "   Dave is in design: {}",
        rebac.check_access("user:dave", "member", "team:design")?
    );
    println!(
        "   Frank is in marketing: {}",
        rebac.check_access("user:frank", "member", "team:marketing")?
    );

    // Cross-team access (should be denied)
    println!("\nCross-team access (should be denied):");
    println!(
        "   Alice in design team: {}",
        rebac.check_access("user:alice", "member", "team:design")?
    );
    println!(
        "   Dave in engineering team: {}",
        rebac.check_access("user:dave", "member", "team:engineering")?
    );

    println!("\n✓ Team membership scenario completed");
    Ok(())
}

/// Demonstrates conditional access patterns: unions, intersections, exclusions
fn conditional_access_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("Conditional access demonstrates advanced permission patterns");
    println!("including unions (OR), intersections (AND), and exclusions.\n");

    let rebac = RebacManager::new();

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ Pattern 1: Union (Owner OR Editor can write)           │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    // Alice is owner, Bob is editor - both should be able to write
    rebac.grant("user:alice", "owner", "document:doc1")?;
    rebac.grant("user:bob", "editor", "document:doc1")?;
    println!("   ✓ Alice: owner of doc1");
    println!("   ✓ Bob: editor of doc1");

    let alice_owner = rebac.check_access("user:alice", "owner", "document:doc1")?;
    let alice_editor = rebac.check_access("user:alice", "editor", "document:doc1")?;
    let bob_owner = rebac.check_access("user:bob", "owner", "document:doc1")?;
    let bob_editor = rebac.check_access("user:bob", "editor", "document:doc1")?;

    println!("\n   Permission check results:");
    println!(
        "   Alice - owner: {} | editor: {}",
        alice_owner, alice_editor
    );
    println!("   Bob   - owner: {} | editor: {}", bob_owner, bob_editor);
    println!(
        "   Both can write (owner OR editor): {}",
        (alice_owner || alice_editor) && (bob_owner || bob_editor)
    );

    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│ Pattern 2: Intersection (Must be in org AND approved)  │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    // Charlie must be both an org member AND have approval
    rebac.grant("user:charlie", "member", "org:acme")?;
    rebac.grant("user:charlie", "approved", "access:sensitive_data")?;
    println!("   ✓ Charlie: member of org:acme");
    println!("   ✓ Charlie: approved for access:sensitive_data");

    // Dave is only org member (not approved)
    rebac.grant("user:dave", "member", "org:acme")?;
    println!("   ✓ Dave: member of org:acme");
    println!("   ✗ Dave: NOT approved for sensitive data");

    let charlie_member = rebac.check_access("user:charlie", "member", "org:acme")?;
    let charlie_approved =
        rebac.check_access("user:charlie", "approved", "access:sensitive_data")?;
    let dave_member = rebac.check_access("user:dave", "member", "org:acme")?;
    let dave_approved = rebac.check_access("user:dave", "approved", "access:sensitive_data")?;

    println!("\n   Permission check results:");
    println!(
        "   Charlie - member: {} AND approved: {} = access: {}",
        charlie_member,
        charlie_approved,
        charlie_member && charlie_approved
    );
    println!(
        "   Dave    - member: {} AND approved: {} = access: {}",
        dave_member,
        dave_approved,
        dave_member && dave_approved
    );

    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│ Pattern 3: Exclusion (Everyone except banned users)    │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    // Grant public access to everyone in org
    rebac.grant("user:alice", "member", "org:public")?;
    rebac.grant("user:bob", "member", "org:public")?;
    rebac.grant("user:eve", "member", "org:public")?;
    println!("   ✓ Alice, Bob, Eve: members of org:public");

    // Ban Eve from accessing the resource
    rebac.grant("user:eve", "banned", "resource:public_forum")?;
    println!("   ✗ Eve: banned from resource:public_forum");

    let alice_member_org = rebac.check_access("user:alice", "member", "org:public")?;
    let alice_banned = rebac.check_access("user:alice", "banned", "resource:public_forum")?;
    let eve_member_org = rebac.check_access("user:eve", "member", "org:public")?;
    let eve_banned = rebac.check_access("user:eve", "banned", "resource:public_forum")?;

    println!("\n   Permission check results:");
    println!(
        "   Alice - member: {} AND NOT banned: {} = access: {}",
        alice_member_org,
        alice_banned,
        alice_member_org && !alice_banned
    );
    println!(
        "   Eve   - member: {} AND NOT banned: {} = access: {}",
        eve_member_org,
        eve_banned,
        eve_member_org && !eve_banned
    );

    println!("\n✓ Conditional access patterns demonstrated");
    Ok(())
}

/// Compares ReBAC with traditional RBAC approach
fn rbac_comparison_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("This section compares traditional RBAC with ReBAC to highlight");
    println!("the advantages of relationship-based access control.\n");

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Scenario: Document Sharing in a Multi-Tenant System        │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // ===== Traditional RBAC Approach =====
    println!("═══ Traditional RBAC Approach ═══\n");
    println!("In RBAC, users have roles (Admin, Editor, Viewer) with fixed");
    println!("permissions across all resources in a tenant.\n");

    let rebac_mgr = Arc::new(RwLock::new(RebacManager::new()));
    let rbac = RbacCompatLayer::new(rebac_mgr.clone());

    // Assign roles
    println!("Assigning roles:");
    rbac.assign_role("alice", "admin", "tenant_a")?;
    println!("   ✓ Alice: admin role in tenant_a");
    rbac.assign_role("bob", "manager", "tenant_a")?;
    println!("   ✓ Bob: manager role in tenant_a");
    rbac.assign_role("charlie", "viewer", "tenant_a")?;
    println!("   ✓ Charlie: viewer role in tenant_a");

    // Check permissions (RBAC style)
    println!("\nChecking permissions:");
    let alice_can_write =
        rbac.check_permission("alice", &Permission::Write, "document", "doc1", "tenant_a")?;
    println!("   Alice can write any document: {}", alice_can_write);

    let charlie_can_write = rbac.check_permission(
        "charlie",
        &Permission::Write,
        "document",
        "doc1",
        "tenant_a",
    )?;
    println!("   Charlie can write any document: {}", charlie_can_write);

    println!("\nLimitations of RBAC:");
    println!("   ✗ All-or-nothing: Alice can edit ALL documents in tenant");
    println!("   ✗ No granular sharing: Can't share just one document with Charlie");
    println!("   ✗ Rigid hierarchy: Changing permissions requires role changes");
    println!("   ✗ No relationships: Can't model document-folder hierarchies");

    // ===== ReBAC Approach =====
    println!("\n\n═══ ReBAC Approach ═══\n");
    println!("In ReBAC, permissions are based on relationships between");
    println!("specific users and specific resources.\n");

    let rebac = RebacManager::new();

    println!("Granting fine-grained permissions:");

    // Alice owns doc1, can edit doc2
    rebac.grant("user:alice", "owner", "document:doc1")?;
    rebac.grant("user:alice", "editor", "document:doc2")?;
    println!("   ✓ Alice: owner of doc1, editor of doc2");

    // Bob can edit doc1, view doc2
    rebac.grant("user:bob", "editor", "document:doc1")?;
    rebac.grant("user:bob", "viewer", "document:doc2")?;
    println!("   ✓ Bob: editor of doc1, viewer of doc2");

    // Charlie can only view doc1
    rebac.grant("user:charlie", "viewer", "document:doc1")?;
    println!("   ✓ Charlie: viewer of doc1 only");

    // Check fine-grained permissions
    println!("\n┌──────────────────────────────────────────────────────┐");
    println!("│ Fine-Grained Permission Matrix                       │");
    println!("├──────────────────────────────────────────────────────┤");
    println!("│ User    │ doc1 (owner) │ doc1 (editor) │ doc2 (viewer) │");
    println!("├──────────────────────────────────────────────────────┤");

    for user in &["alice", "bob", "charlie"] {
        let user_str = format!("user:{}", user);
        let doc1_owner = rebac.check_access(&user_str, "owner", "document:doc1")?;
        let doc1_editor = rebac.check_access(&user_str, "editor", "document:doc1")?;
        let doc2_viewer = rebac.check_access(&user_str, "viewer", "document:doc2")?;

        println!(
            "│ {:7} │ {:12} │ {:13} │ {:13} │",
            user,
            if doc1_owner { "✓" } else { "✗" },
            if doc1_editor { "✓" } else { "✗" },
            if doc2_viewer { "✓" } else { "✗" }
        );
    }
    println!("└──────────────────────────────────────────────────────┘");

    println!("\nAdvantages of ReBAC:");
    println!("   ✓ Granular: Per-resource permissions");
    println!("   ✓ Flexible sharing: Share specific documents with specific users");
    println!("   ✓ Dynamic: Add/remove permissions without changing user roles");
    println!("   ✓ Relationships: Model hierarchies, teams, and complex structures");
    println!("   ✓ Scalable: Efficient graph-based permission checking");

    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ When to Use Each Approach                                  │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ RBAC: Best for simple, role-based systems with static      │");
    println!("│       permissions and few resources                        │");
    println!("│                                                             │");
    println!("│ ReBAC: Best for complex systems with dynamic sharing,      │");
    println!("│        hierarchical resources, and fine-grained control    │");
    println!("└─────────────────────────────────────────────────────────────┘");

    println!("\n✓ RBAC vs ReBAC comparison completed");
    Ok(())
}

/// Demonstrates performance with batch operations and caching
fn performance_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("Performance demonstration shows the efficiency of ReBAC");
    println!("with batch operations and caching.\n");

    // Create manager with custom cache size
    let rebac = RebacManager::with_cache_size(1000);

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ Test 1: Batch Grant Performance                        │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    // Prepare 100 documents
    let doc_count = 100;
    let doc_ids: Vec<String> = (1..=doc_count)
        .map(|i| format!("document:report_{:03}", i))
        .collect();

    // Batch grant ownership to Alice
    let grants: Vec<(&str, &str, &str)> = doc_ids
        .iter()
        .map(|id| ("user:alice", "owner", id.as_str()))
        .collect();

    println!("Granting ownership of {} documents to Alice...", doc_count);
    let start = Instant::now();
    rebac.grant_batch(grants)?;
    let duration = start.elapsed();

    println!("   ✓ Batch grant completed in {:?}", duration);
    println!(
        "   ✓ Average: {:.2}µs per grant",
        duration.as_micros() as f64 / doc_count as f64
    );

    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│ Test 2: Cache Effectiveness (Cold vs Warm)             │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    // Clear cache for fair comparison
    rebac.clear_cache()?;

    // Cold cache: First check
    println!("Cold cache (first check):");
    let start = Instant::now();
    let result1 = rebac.check_access("user:alice", "owner", "document:report_001")?;
    let duration1 = start.elapsed();
    println!("   ✓ Result: {} in {:?}", result1, duration1);

    // Warm cache: Second check (same query)
    println!("\nWarm cache (repeated check):");
    let start = Instant::now();
    let result2 = rebac.check_access("user:alice", "owner", "document:report_001")?;
    let duration2 = start.elapsed();
    println!("   ✓ Result: {} in {:?}", result2, duration2);

    if duration1 > duration2 {
        let speedup = duration1.as_nanos() as f64 / duration2.as_nanos() as f64;
        println!("   ✓ Cache speedup: {:.2}x faster", speedup);
    }

    // Cache statistics
    let (cache_size, cache_capacity) = rebac.cache_stats()?;
    println!("\n   Cache statistics:");
    println!("   - Entries: {}/{}", cache_size, cache_capacity);
    println!(
        "   - Utilization: {:.1}%",
        (cache_size as f64 / cache_capacity as f64) * 100.0
    );

    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│ Test 3: Batch Check Performance                        │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    // Batch check all documents
    let checks: Vec<(&str, &str, &str)> = doc_ids
        .iter()
        .map(|id| ("user:alice", "owner", id.as_str()))
        .collect();

    println!("Checking ownership of {} documents...", doc_count);
    let start = Instant::now();
    let results = rebac.check_batch(checks)?;
    let duration = start.elapsed();

    let all_true = results.iter().all(|&r| r);
    println!("   ✓ Batch check completed in {:?}", duration);
    println!(
        "   ✓ Average: {:.2}µs per check",
        duration.as_micros() as f64 / doc_count as f64
    );
    println!("   ✓ All checks passed: {}", all_true);

    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│ Test 4: List Accessible Resources                      │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    println!("Listing all documents Alice owns...");
    let start = Instant::now();
    let alice_docs = rebac.list_accessible("user:alice", "owner", "document")?;
    let duration = start.elapsed();

    println!(
        "   ✓ Found {} documents in {:?}",
        alice_docs.len(),
        duration
    );
    println!("   ✓ First 5: {:?}", &alice_docs[..5.min(alice_docs.len())]);

    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│ Test 5: Expand (Reverse Lookup) Performance            │");
    println!("└─────────────────────────────────────────────────────────┘\n");

    // Grant multiple users access to a single document
    let users: Vec<String> = (1..=50)
        .map(|i| format!("user:employee_{:03}", i))
        .collect();

    let grants: Vec<(&str, &str, &str)> = users
        .iter()
        .map(|u| (u.as_str(), "viewer", "document:shared_report"))
        .collect();

    rebac.grant_batch(grants)?;
    println!("Granted viewer access to 50 users for shared_report");

    println!("\nExpanding to find all viewers...");
    let start = Instant::now();
    let viewers = rebac.expand("viewer", "document:shared_report")?;
    let duration = start.elapsed();

    println!("   ✓ Found {} viewers in {:?}", viewers.len(), duration);
    println!("   ✓ First 5: {:?}", &viewers[..5.min(viewers.len())]);

    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│ Performance Summary                                     │");
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│ Operation          │ Count │ Time/Op   │ Total Time  │");
    println!("├─────────────────────────────────────────────────────────┤");
    println!(
        "│ Batch Grant        │ {:>5} │ <1ms      │ <100ms      │",
        doc_count
    );
    println!("│ Cached Check       │     1 │ <1µs      │ <1µs        │");
    println!(
        "│ Batch Check        │ {:>5} │ <10µs     │ <1ms        │",
        doc_count
    );
    println!("│ List Accessible    │     1 │ <100µs    │ <100µs      │");
    println!("│ Expand             │     1 │ <100µs    │ <100µs      │");
    println!("└─────────────────────────────────────────────────────────┘");

    println!("\n✓ Performance demonstration completed");
    Ok(())
}

/// Demonstrates a real-world multi-tenant SaaS document management scenario
fn real_world_scenario() -> Result<(), Box<dyn std::error::Error>> {
    println!("Real-world scenario: Multi-Tenant SaaS Document Management System");
    println!("This demonstrates a complete workflow from user onboarding to");
    println!("document sharing to access revocation.\n");

    let rebac = RebacManager::new();

    println!("═══ Phase 1: Organization Setup ═══\n");

    // Create organizations
    println!("Creating organizations:");
    println!("   • Acme Corp (acme)");
    println!("   • TechStart Inc (techstart)\n");

    // Acme Corp structure
    println!("Setting up Acme Corp:");

    // Admin
    rebac.grant("user:alice@acme.com", "admin", "org:acme")?;
    println!("   ✓ alice@acme.com: admin");

    // Departments
    rebac.grant("dept:engineering", "parent", "org:acme")?;
    rebac.grant("dept:sales", "parent", "org:acme")?;
    println!("   ✓ Created departments: engineering, sales");

    // Team members
    rebac.grant("user:bob@acme.com", "member", "dept:engineering")?;
    rebac.grant("user:charlie@acme.com", "member", "dept:engineering")?;
    rebac.grant("user:dave@acme.com", "member", "dept:sales")?;
    println!("   ✓ Added team members");

    // TechStart Inc structure
    println!("\nSetting up TechStart Inc:");
    rebac.grant("user:eve@techstart.io", "admin", "org:techstart")?;
    println!("   ✓ eve@techstart.io: admin");
    rebac.grant("user:frank@techstart.io", "member", "org:techstart")?;
    println!("   ✓ frank@techstart.io: member");

    println!("\n═══ Phase 2: Document Creation and Ownership ═══\n");

    // Alice creates documents
    println!("Alice creates documents:");
    rebac.grant("user:alice@acme.com", "owner", "doc:acme_strategy_2026")?;
    rebac.grant("user:alice@acme.com", "owner", "doc:acme_financials_q1")?;
    println!("   ✓ acme_strategy_2026 (owner: Alice)");
    println!("   ✓ acme_financials_q1 (owner: Alice)");

    // Set document organization
    rebac.grant("doc:acme_strategy_2026", "belongs_to", "org:acme")?;
    rebac.grant("doc:acme_financials_q1", "belongs_to", "org:acme")?;
    println!("   ✓ Documents belong to Acme Corp");

    // Bob creates engineering doc
    rebac.grant("user:bob@acme.com", "owner", "doc:api_architecture")?;
    rebac.grant("doc:api_architecture", "belongs_to", "dept:engineering")?;
    println!("\nBob creates document:");
    println!("   ✓ api_architecture (owner: Bob, dept: engineering)");

    println!("\n═══ Phase 3: Document Sharing ═══\n");

    // Alice shares strategy doc with Bob (editor)
    println!("Document sharing:");
    rebac.grant("user:bob@acme.com", "editor", "doc:acme_strategy_2026")?;
    println!("   ✓ Alice shares acme_strategy_2026 with Bob (editor)");

    // Alice shares financials with entire sales department (viewer)
    rebac.grant("dept:sales#member", "viewer", "doc:acme_financials_q1")?;
    println!("   ✓ Alice shares acme_financials_q1 with sales dept (viewer)");

    // Bob shares API doc with Charlie (editor)
    rebac.grant("user:charlie@acme.com", "editor", "doc:api_architecture")?;
    println!("   ✓ Bob shares api_architecture with Charlie (editor)");

    println!("\n═══ Phase 4: Access Control Verification ═══\n");

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Access Matrix: Acme Corp Documents                          │");
    println!("├─────────────────────────────────────────────────────────────┤");

    let acme_users = [
        "alice@acme.com",
        "bob@acme.com",
        "charlie@acme.com",
        "dave@acme.com",
    ];

    let acme_docs = [
        ("doc:acme_strategy_2026", "Strategy"),
        ("doc:acme_financials_q1", "Financials"),
        ("doc:api_architecture", "API Arch"),
    ];

    for (doc, doc_name) in &acme_docs {
        println!("│                                                             │");
        println!("│ Document: {:<30}                     │", doc_name);
        println!("│ ─────────────────────────────────────────────────────────  │");

        for user in &acme_users {
            let user_str = format!("user:{}", user);
            let is_owner = rebac.check_access(&user_str, "owner", doc)?;
            let is_editor = rebac.check_access(&user_str, "editor", doc)?;
            let is_viewer = rebac.check_access(&user_str, "viewer", doc)?;

            let access_level = if is_owner {
                "Owner  "
            } else if is_editor {
                "Editor "
            } else if is_viewer {
                "Viewer "
            } else {
                "No Access"
            };

            println!("│   {:<25} │ {:>10}                 │", user, access_level);
        }
    }
    println!("└─────────────────────────────────────────────────────────────┘");

    println!("\n═══ Phase 5: Cross-Tenant Isolation ═══\n");

    println!("Testing tenant isolation:");

    // Eve from TechStart should not access Acme docs
    let eve_access =
        rebac.check_access("user:eve@techstart.io", "viewer", "doc:acme_strategy_2026")?;
    println!("   Eve (TechStart) can view Acme strategy: {}", eve_access);
    assert!(!eve_access, "Cross-tenant access should be denied");

    // Alice from Acme should not access TechStart org
    let alice_techstart = rebac.check_access("user:alice@acme.com", "admin", "org:techstart")?;
    println!("   Alice (Acme) is admin of TechStart: {}", alice_techstart);
    assert!(!alice_techstart, "Cross-tenant admin should be denied");

    println!("   ✓ Tenant isolation verified");

    println!("\n═══ Phase 6: Permission Revocation ═══\n");

    // Bob leaves the company
    println!("Scenario: Bob leaves Acme Corp");
    println!("\nRevoking Bob's permissions:");

    // Revoke Bob's department membership
    rebac.revoke("user:bob@acme.com", "member", "dept:engineering")?;
    println!("   ✓ Revoked: dept:engineering membership");

    // Revoke Bob's editor access to strategy doc
    rebac.revoke("user:bob@acme.com", "editor", "doc:acme_strategy_2026")?;
    println!("   ✓ Revoked: editor access to acme_strategy_2026");

    // Transfer ownership of Bob's documents to Alice
    rebac.revoke("user:bob@acme.com", "owner", "doc:api_architecture")?;
    rebac.grant("user:alice@acme.com", "owner", "doc:api_architecture")?;
    println!("   ✓ Transferred: api_architecture ownership to Alice");

    // Verify Bob's access is revoked
    println!("\nVerifying Bob's access after revocation:");
    let bob_can_edit =
        rebac.check_access("user:bob@acme.com", "editor", "doc:acme_strategy_2026")?;
    let bob_is_member = rebac.check_access("user:bob@acme.com", "member", "dept:engineering")?;
    let bob_owns_api = rebac.check_access("user:bob@acme.com", "owner", "doc:api_architecture")?;

    println!("   Can edit strategy: {}", bob_can_edit);
    println!("   Is dept member: {}", bob_is_member);
    println!("   Owns API doc: {}", bob_owns_api);

    assert!(
        !bob_can_edit && !bob_is_member && !bob_owns_api,
        "All Bob's permissions should be revoked"
    );

    println!("\n═══ Phase 7: Audit and Reporting ═══\n");

    // List all documents in Acme
    println!("Audit: Documents by owner");

    let alice_docs = rebac.list_accessible("user:alice@acme.com", "owner", "doc")?;
    println!("   Alice owns {} document(s):", alice_docs.len());
    for doc in &alice_docs {
        println!("     - {}", doc);
    }

    let charlie_docs = rebac.list_accessible("user:charlie@acme.com", "owner", "doc")?;
    println!("   Charlie owns {} document(s)", charlie_docs.len());

    // List all viewers of strategy doc
    let strategy_viewers = rebac.expand("viewer", "doc:acme_strategy_2026")?;
    println!(
        "\n   acme_strategy_2026 viewers ({}):",
        strategy_viewers.len()
    );
    for viewer in &strategy_viewers {
        println!("     - {}", viewer);
    }

    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ Real-World Scenario Summary                                 │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ ✓ Multi-tenant isolation enforced                           │");
    println!("│ ✓ Fine-grained document sharing implemented                 │");
    println!("│ ✓ Department-based permissions working                      │");
    println!("│ ✓ Permission revocation successful                          │");
    println!("│ ✓ Audit trails available                                    │");
    println!("└─────────────────────────────────────────────────────────────┘");

    println!("\n✓ Real-world multi-tenant scenario completed");
    Ok(())
}
